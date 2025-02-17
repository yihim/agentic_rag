from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from sentence_transformers import SentenceTransformer
from agents.constants.models import (
    EMBEDDING_MODEL,
    TABLE_ORGANIZER_LLM,
    TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
    TABLE_ORGANIZER_LLM_MAX_TOKENS,
    VISION_LLM,
    VISION_LLM_SYSTEM_PROMPT,
    VISION_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM,
    AGENTIC_CHUNKER_LLM_MAX_TOKENS,
    AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
)
import re
import requests
from tqdm import tqdm
import torch
import shutil
import string
from time import perf_counter
from llm_preprocess_data import process_data, process_image_data
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


def clean_and_organize_external_data(file_name_without_ext: str):

    image_dir = "../../data_extraction/tmp/images/"
    md_dir = "../../data_extraction/tmp/md/"

    def clean_text_to_json(text):
        # Split the text into lines
        lines = text.split("\n")

        # Initialize variables
        json_output = []
        current_header = None
        current_content = []
        non_header_content = []

        def clean_header(header):
            # Remove the # symbol and any leading/trailing whitespace
            header = header.replace("#", "").strip()
            # Remove any numbers and special characters, keeping only letters and spaces
            header = re.sub(r"[^a-zA-Z\s]", "", header)
            # Remove extra spaces and strip
            header = " ".join(header.split())
            return header

        def process_current_group():
            if current_header:
                # Check if content ends with punctuation or contains .jpg
                has_valid_content = any(
                    line.strip()[-1] in string.punctuation or ".jpg" in line
                    for line in current_content
                )

                # Only add the group if there's content and it meets our criteria
                if current_content and has_valid_content:
                    json_output.append(
                        {
                            "header": clean_header(current_header),
                            "content": "\n".join(current_content),
                        }
                    )
                elif not current_content:
                    # If header has no content, add it with empty content
                    json_output.append(
                        {"header": clean_header(current_header), "content": ""}
                    )

        # Process each line
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Check if line is a header
            if line.startswith("#"):
                # Process previous group before starting new one
                process_current_group()
                # Start new group
                current_header = line
                current_content = []
            else:
                if current_header:
                    # Add line to current header's content
                    current_content.append(line)
                else:
                    # Add line to non-header content
                    non_header_content.append(line)

        # Process the last group
        process_current_group()

        # Add non-header content if it exists and has valid content
        if non_header_content and any(
            line.strip()[-1] in string.punctuation or ".jpg" in line
            for line in non_header_content
        ):
            json_output.append(
                {"header": "None", "content": "\n".join(non_header_content)}
            )

        # Convert to JSON with ensure_ascii=False to preserve Unicode characters
        return json_output

    def clean_references(text):
        # Pattern to match:
        # 1. Period followed by numbers and commas
        # 2. Comma followed by numbers
        # Both patterns should be followed by a space or end of string
        pattern = r"[.,][0-9]+(?:,\s*[0-9]+)*(?=\s|$)"

        # Function to process each match
        def replace_match(match):
            # If match starts with period, return period
            if match.group().startswith("."):
                return "."
            # If match starts with comma, return comma
            elif match.group().startswith(","):
                return ","

        # Replace the pattern
        cleaned_text = re.sub(pattern, replace_match, text)

        return cleaned_text.strip()

    with open(
        os.path.join(md_dir, file_name_without_ext + ".md"), "r", encoding="utf-8"
    ) as f:
        extracted_data = f.read()

    print("Cleaning extracted data...")

    cleaned_references = clean_references(extracted_data)

    organized_data = clean_text_to_json(cleaned_references)

    print("Cleaning completed.")

    text_data = []
    table_data = []
    image_data = []
    organized_all_data = []

    def extract_image_file_name(text: str) -> str:
        match = re.search(r"/([^/]+\.[a-zA-Z0-9]+)\)", text)
        return match.group(1)

    # Organize data
    for index, data in enumerate(organized_data):
        header = data["header"]
        content = data["content"].split("\n")

        text = []

        for item in content:
            if item:
                if item.startswith("<html><body><table>"):
                    table_data.append({"header": header, "table": item})
                elif item.startswith("![]"):
                    image_data.append(
                        {"header": header, "image": extract_image_file_name(item)}
                    )
                else:
                    text.append(item)

        if text:
            text_data.append({"header": header, "text": text})

    # Process table data
    if table_data:
        table_data = process_data(
            data=table_data,
            data_type="table",
            llm_name=TABLE_ORGANIZER_LLM,
            system_prompt=TABLE_ORGANIZER_LLM_SYSTEM_PROMPT,
            max_tokens=TABLE_ORGANIZER_LLM_MAX_TOKENS,
            batch_size=8,
        )

        # print(table_data)

        organized_all_data.extend(table_data)

    # Process image data
    if image_data:
        image_dir = os.path.join(image_dir, file_name_without_ext)
        image_data = process_image_data(
            data=image_data,
            llm_name=VISION_LLM,
            system_prompt=VISION_LLM_SYSTEM_PROMPT,
            max_tokens=VISION_LLM_MAX_TOKENS,
            image_dir=image_dir,
        )

        # print(image_data)

        organized_all_data.extend(image_data)

    # Process text data
    if text_data:
        text_data = process_data(
            data=text_data,
            data_type="text",
            llm_name=AGENTIC_CHUNKER_LLM,
            system_prompt=AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT,
            max_tokens=AGENTIC_CHUNKER_LLM_MAX_TOKENS,
            batch_size=12,
        )

        # print(text_data)

        organized_all_data.extend(text_data)

    return organized_all_data


def extract_data_from_source(data_path: str):
    file_name = os.path.basename(data_path)
    file_name_without_ext = file_name.split(".")[0]
    mounted_dir = "../../data_extraction/data/"
    shutil.copy(data_path, mounted_dir)
    print("Extracting data from source...")
    response = requests.post(
        "http://localhost:8000/extract", json={"file_path": f"./data/{file_name}"}
    )
    return response.status_code, file_name_without_ext


def save_vector_to_store(path: str, data_path: str):
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    # response_status, file_name_without_ext = extract_data_from_source(data_path)
    # if response_status == 200:
    #     print("Data extraction completed successfully.")
    #     organized_all_data = clean_and_organize_external_data(file_name_without_ext)
    #     print("All data processed successfully.")

    # organized_all_data =  clean_and_organize_external_data("2a85b52768ea5761b773be49b09d15f0b95415b0")
    # print(organized_all_data)

    organized_all_data = [{'header': 'Predictive power of conflict history', 'table': '<html><body><table><tr><td></td><td>WPS model</td><td>Simple model</td></tr><tr><td>Recall</td><td>0.86</td><td>0.71</td></tr><tr><td>Precision</td><td>0.47</td><td>0.73</td></tr><tr><td>F2</td><td>0.74</td><td>0.71</td></tr><tr><td>ROCAUC</td><td>0.89</td><td>0.84</td></tr><tr><td>AUPRC</td><td>0.42</td><td>0.55</td></tr><tr><td>Brier score</td><td>0.084</td><td>0.057</td></tr></table></body></html>', 'text': 'This table compares performance metrics of two models, the WPS model and a Simple model, in a machine learning context. The metrics include Recall, Precision, F2 score, ROCAUC, AUPRC, and Brier score. The WPS model generally outperforms the Simple model across most metrics, except for the Brier score, where the Simple model has a lower (better) score.'}, {'header': 'ASSESSING THE TECHNICAL FEASIBILITY OF CONFLICT PREDICTION FOR ANTICIPATORY ACTION', 'image': 'b43d7bf42f48cac173b89ae53212a13cfbca344c3639cd42063536dd31bc29b6.jpg', 'text': 'The image depicts a line graph set against a teal background, featuring a pattern of small white dots. The graph\'s x-axis and y-axis are not labeled, but it appears to display a trend over time, with the line beginning in the lower left and rising to a peak before descending to the right. The line is composed of red and white dots, with the red dots forming the majority of the line. The graph is dated "October 2022" in the upper left corner.'}, {'header': 'Predictive power of conflict history', 'image': '3a83e56ed1e648a3c36de0fd20ff59700dbec417ac6d81c80d5e2fb84b9a5bc0.jpg', 'text': 'The image presents a line graph illustrating the relationship between precision and recall, with a focus on the years 01-09 and 10-18. The graph features two lines, one in red and the other in blue, which represent the data for the specified time periods. The x-axis is labeled "Recall" and ranges from 0.0 to 1.0, while the y-axis is labeled "Precision" and spans from 0.0 to 1.0. The graph\'s background is white, providing a clear visual representation of the data.'}, {'header': 'ACKNOWLEDGEMENTS', 'text': ['This review was undertaken by the United Nations Office for the Coordination of Humanitarian Affairs (OCHA).', 'OCHA is located in The Hague.', 'The Centre for Humanitarian Data is part of OCHA.', 'The study was written by Seth Caldwell.', 'Håvard Hegre, Kim Kristensen, Erin Lentz, Leonardo Milano, Ben Parker, Josée Poirier, Manu Singh, Sarah Telford, and Marie Wagner provided internal and external review.', 'Graphic design for the study was provided by Lena Kim.', 'The Centre for Humanitarian Data can be contacted at centrehumdata@un.org.']}, {'header': 'EXECUTIVE SUMMARY', 'text': ['Anticipatory action helps humanitarian organizations prepare for predictable shocks to reduce their impact on vulnerable people.', "The Centre for Humanitarian Data has supported OCHA's anticipatory action frameworks in over a dozen countries since 2020.", 'The focus of this support is on developing trigger mechanisms for releasing funds and taking action before a projected shock.', 'Current anticipatory action pilots have successfully predicted climate events and disease outbreaks using data and models.', 'OCHA leadership and stakeholders have inquired about predicting and acting ahead of conflicts using similar techniques.', 'Conflict prediction has been a goal for social studies researchers for a generation.', 'A wide array of literature from various research projects was reviewed to assess the feasibility of conflict prediction.', 'Advances in machine learning and new historical datasets have given momentum to conflict prediction research.', 'Conflict prediction remains a complex and challenging problem.', 'Three types of conflict prediction models were evaluated: classification, risk prediction, and continuous prediction.', 'There is insufficient justification for relying exclusively on conflict prediction models for anticipatory action due to poor performance, lack of clear connection to humanitarian impact, and the dominance of ongoing conflict as a predictor.', 'To use conflict prediction for anticipatory action, future work should utilize flexible models, focus on predicting shifts in conflicts, incorporate human inputs, improve predictions on humanitarian impact, ensure reproducibility and transparency, and learn from academic research.', 'The goal is to make applied research relevant for humanitarian decision-making.']}, {'header': 'MOTIVATION', 'text': ['Anticipatory action enables humanitarian organizations to prepare for predictable shocks to reduce their impact on vulnerable people.', 'The Centre for Humanitarian Data has supported OCHA’s anticipatory action frameworks in over a dozen countries since 2020.', 'The work focuses on developing trigger mechanisms for releasing funds and taking action before a projected shock.', 'Current anticipatory action pilots have used data and models to predict climate events and disease outbreaks.', 'Conflict is a key driver of food insecurity globally, affecting nearly 100 million people in 23 conflict-affected countries in 2020.', 'In 2021, 36 armed conflicts were reported, indicating ongoing conflict-driven humanitarian needs.', 'OCHA leadership and stakeholders have inquired about the feasibility of using similar techniques to predict and act ahead of conflicts.']}, {'header': 'APPROACH', 'text': ["The Centre's research focused on answering two questions: How accurate are conflict forecasts? How well can various sources predict different types of conflict in specific situations?", 'Conflict does not appear out of nowhere.', 'Politics, the environment, or competition for resources may all contribute to a flare-up of violence.', 'Conflict prediction generally relies on an analysis of historical conflict and contributing factors.', 'Conflict prediction aims to build a model of where and when conflict may break out in the future.', 'The range of factors and differences across contexts can make it challenging for any data-driven model to accurately predict conflict.', 'The Centre reviewed existing literature and models to see how well they performed in predicting conflict and the feasibility of applying these models to anticipatory action.', 'The paper defines and evaluates three types of conflict prediction models: classification, risk prediction, and continuous prediction.', 'The paper concludes with a set of recommendations and next steps for the humanitarian sector.']}, {'header': 'TYPES OF PREDICTION', 'text': ['The literature is dominated by models that fall under three types of conflict prediction: classification, risk prediction, and continuous prediction.', 'Classification models categorically predict whether or not a conflict will occur in a particular area and time.', 'Classification models use binary (e.g., yes/no) or multiclass classification (e.g., major, minor or no conflict, or a 1-5 scale).', 'Risk prediction is designed to generate a measure of underlying risk of conflict, usually produced as a probability of conflict.', 'Risk prediction is usually given as a probability of conflict at a certain geographic scale within a certain timeframe.', 'Continuous prediction models directly predict a specific measure of conflict, such as the number of fatalities or conflict events.', 'Continuous prediction models do not categorize or scale the results.', "For example, if predicting fatalities due to state-based violence in Mali in February 2022, a classification model might predict 'There will be conflict.'", "A risk prediction model might predict 'There is a 75 percent probability of conflict.'", "A continuous prediction model might predict 'There will be 33 fatalities due to conflict.'", 'Conflict is defined as occurring if there are 25 or more conflict-related fatalities in one month in Mali.', 'Further details on each of these models are available in the Technical Annex.']}, {'header': 'MODELS IN THE HUMANITARIAN SECTOR', 'text': ["OCHA's anticipatory action frameworks use classification models in the Philippines, Bangladesh, and Somalia/Ethiopia to assess hazards and shocks.", "OCHA's frameworks in the Philippines rely on classification models to assess typhoons.", "OCHA's frameworks in Bangladesh rely on classification models to assess floods.", "OCHA's frameworks in Somalia and Ethiopia rely on classification models to assess droughts.", "Classification models in OCHA's frameworks set thresholds on probabilities of events.", 'In the Philippines, if there is over 50 percent probability of 80 percent of houses being damaged by a typhoon, anticipatory action is triggered.', 'The Integrated Food Security Phase Classification (IPC) is a classification model in the humanitarian sector.', 'IPC generates five classes representing the projected phase of acute food insecurity, with five being the most severe.', 'The INFORM Risk Index is a risk prediction model that measures the general risk of crisis for a country based on structural factors.', 'INFORM uses the Global Conflict Risk Index as one of its key inputs.', 'The global displacement forecasts produced by the Danish Refugee Council are continuous predictions of the number of displaced persons.']}, {'header': 'MODEL ASPECTS', 'text': ['For these three models types, forecasts of conflict are defined by parameters.', 'Lead time is how far in advance the model predicts conflict.', 'Lead time can range from one month to often between one and three years, but can go out to 50 years.', 'Length of forecasting period is the temporal range of the forecast period.', 'The length of forecast is typically one month but can go up to a ten-year period.', 'Lead time and length of forecasting period are different.', 'Geographic distribution is the geographic scope of the prediction exercise.', 'Geographic distribution is almost always at the country level or more granular.', 'Type refers to the type of conflict being predicted, such as state vs. non-state actors.', 'Scale refers to the scale of the conflict being predicted, often defined in terms of number of deaths.', 'Scale can define whether a conflict is happening or not, or can be more complex.', 'Scale is only used in classification and risk prediction models.', 'Continuous predictions do not use a definition of scale.']}, {'header': 'OVERALL PERFORMANCE', 'text': ['There are common issues affecting the usability of all three models.', 'The scale of predicted conflict is a common issue.', 'Predicting conflict onset and escalation is a common issue.', 'The lack of linkages between predicted conflict and humanitarian impact is a common issue.', 'The lack of linkages between predicted conflict and humanitarian impact is critically important to the feasibility of applying these models for anticipatory action.', 'Details on the feasibility of each model are available in the Technical Annex.']}, {'header': 'PREDICTED CONFLICT AND HUMANITARIAN IMPACT', 'text': ['Models often predict conflict at a small scale in terms of casualties or fatalities.', 'These predictions cover a large timeline and a wide geographic area.', 'Academic models can perform well, but the scale can be as small as predicting a single conflict fatality in a given month and country.', 'The humanitarian impact of one fatality in a month or 10 conflict deaths in a year in a country is not readily derivable from the models.', 'Defining conflict onset is an issue where onset is typically defined as the first time when the scale of conflict used for the model is observed.', 'Conflict onset is defined as the month with 25 battle-related deaths after 24 preceding months without 25 battle-related deaths.', 'Even if conflict prediction improves to forecast conflict escalation, it is not clear that this would directly predict the dynamics of humanitarian needs.', 'Predicting the humanitarian impact of a conflict remains a challenge.', 'This issue is not just for conflict prediction but also for other applications of anticipatory action, such as for climate hazards.', 'Research on the dynamics of conflict and its potential impact on humanitarian needs and response requires more investigation.', 'This research could build on existing work in the food insecurity space, where FEWS NET and the IPC work on integrating multiple data sources to project future levels of food insecurity.', 'New models are being explored to more accurately predict transitions between IPC phases to improve early warning.', 'A new six-year research programme into the complex impacts of armed conflict began in May 2022.']}, {'header': 'ANTICIPATORY ACTION WITHOUT THRESHOLDS', 'text': ['Anticipatory action is typically defined by models with clear thresholds set in a transparent framework.', 'Risk prediction models can perform well, but classification models may not.', 'Classification models are required in an anticipatory action framework.', 'Risk models are typically used for disaster risk reduction, peacekeeping, or security.', 'Risk models may not be immediately applicable to anticipatory action.', 'Further work is needed to explore the use of risk or continuous prediction models in anticipatory action.', 'Classification models may be technically infeasible due to poor performance.']}, {'header': 'RECOMMENDATIONS FOR FUTURE WORK', 'text': ['Given our findings, we do not see immediate applications of conflict prediction for triggering anticipatory action.', 'These recommendations do not preclude the application of anticipatory action in response to other shocks in a conflict setting.', 'The following three areas should be considered as potential avenues towards feasibility:', '1) Adopt modeling best practices.', '2) Explore areas that are under-researched.', '3) Practice transparent development and evaluation of models.']}, {'header': 'ADOPT MODELING BEST PRACTICES', 'text': ['The modeling and prediction of conflict should learn from recent research findings.', 'The research community has identified many predictors of conflict in models.', 'Ensemble methods should be used in conflict models.', 'A vast array of conflict predictors have been identified in the literature.', 'Conflict predictors are not consistently identified across all models.', 'The drivers of conflict vary across political, socioeconomic, and environmental landscapes.', 'Contested elections and political shifts are drivers of conflict.', 'Crop failure and unemployment are drivers of conflict.', 'There is no consensus on the relationship between environmental changes and conflict.', 'Findings are often not robust to changes in the model.', 'Adding new indicators can invalidate claims about predictors and conflict.', 'Identified relationships between predictors and conflict may not be systematic across crises.', 'Similar issues have been identified in linking climate change to conflict.', 'The use of predictive models to validate causal frameworks is increasingly common.', 'Testing how well measures of democratic governance predict a country’s willingness to engage in conflict is an example.', 'The purpose is to justify the importance of a causal factor based on predictive performance.', 'Models focused on a particular driver or theoretical framework often have subpar performance.', 'New work on conflict prediction should recognize conflict as a multi-dimensional problem.', 'Efforts should not focus on strict theoretical frameworks for conflict causality.', 'Efforts should not focus on a single driver, such as the lack of water.', 'These efforts have not proven effective for predicting conflict.']}, {'header': 'Consider ensemble methods', 'text': ['Ensemble methods include random forests, Bayesian model averaging, and gradient boosting machines.', 'These ensemble methods tend to outperform single model specifications in predictive performance.', 'State-of-the-art academic research on conflict prediction focuses almost exclusively on ensemble models.', 'This approach is common in modeling for complex processes such as climate.', 'Future exploratory work in the humanitarian sector should consider ensemble methods.', 'This is particularly important when single model specifications require a lot of time for fine-tuning.', 'Single model specifications may fail to learn different features of the data in ways that ensemble models can.']}, {'header': 'EXPLORE AREAS THAT ARE UNDER RESEARCHED', 'text': ['Based on the literature review, future research should focus on under-researched areas.', 'The recommendation includes utilizing superforecasters and prediction markets.', 'Incorporating local data and analysis is suggested.', 'Predicting shifts in conflict is recommended.', 'Predicting risks, rather than specific events, is advised.', 'Predicting the impact of conflict is recommended.', 'A single model or approach is unlikely to work across all conflicts.', 'Superforecasters and prediction markets have been shown to outperform purely quantitative approaches in certain tasks.', 'Human inputs can identify patterns or quantities that machine learning algorithms might miss.', 'Shifting dynamics of state borders and geopolitics are difficult to capture quantitatively.', 'The utility of purely quantitative predictions of conflict is an open question.', 'Managing a human forecasting network requires significant time and resources.', 'These methods are not common in the research community.', 'Examples often come from the intelligence and defense communities, crowd-sourced systems, and conflict monitoring mechanisms.', 'Some results have been promising, but findings have been questioned.', 'Models using only human inputs or combining human inputs with quantitative data are suggested.']}, {'header': 'Incorporate local data and analysis', 'text': ['Socioeconomic, political, and other input data for conflict prediction are often missing or insufficient, particularly in the most fragile and conflict-affected states.', 'Conflict prediction requires filling in missing inputs.', 'This work is sometimes done directly by conflict modelers.', 'Conflict modelers may implicitly rely on modeled data such as World Bank poverty headcounts or the INFORM Risk Index.', 'Using proxies for conflict prediction presents issues for detecting rapid shifts in conflict dynamics.', 'The poorer quality of proxy data compared with directly measured variables likely limits the overall forecasting potential of quantitative models.', 'Prediction of conflict at the subnational level is typically done by disaggregating data across subnational boundaries or using a grid approach.', 'Using locally captured data may be more sensitive to dynamic shifts that best predict conflict.', 'There is a potential to develop models that predict where conflict will occur, or hotspots of higher risk, rather than when.', 'This approach may be more feasible and still useful for humanitarian operations.', 'Even where local data is available, model performance is often too poor for operationalization.', 'Extensive testing remains a baseline requirement for these models.', 'While these models may improve performance to specific contexts, they will be much more difficult to generalize to a regional or global level.']}, {'header': 'Predict conflict shifts', 'text': ['Recent academic research focuses on predicting the probability of conflict cessation or onset.', 'This requires a more nuanced set of error metrics and validation for models.', 'A recent conflict escalation prediction competition hosted by ViEWS has produced useful metrics for assessing continuous predictions.', 'New developments should focus on linking this work to classification models.', 'These models should assess their performance in predicting large scale conflict onset or escalation.', 'This work aims to support anticipatory action for humanitarian response.', 'It requires the identification and use of time-variant predictors.', 'These predictors should go beyond estimating the underlying risk of conflict.', 'They should better measure when risks are changing due to dynamic factors.', 'This may require the use of more localized or human inputs.', 'The recommendation is to focus on risk prediction models.', 'The use of human forecasters, agent-based modeling, and scenario building could improve conflict risk models.', 'Answering open questions around probabilistic risk prediction is likely simpler than improving classification model performance.', 'Risk models, if well-calibrated, present valuable information for humanitarian response.', 'However, how to apply risk models for anticipatory action remains unclear.']}, {'header': 'Predict the impact of conflict', 'text': ['In response to the war in Ukraine, predictions have proliferated on the impact of the crisis on global food insecurity.', 'The predictions consider factors such as disruption of agricultural production in the region.', 'The predictions also consider disruption to supply chains due to sanctions.', 'The predictions note reduced funding for other crises.', 'The current intensity and interstate nature of the Ukraine conflict make it a relatively unique context for generating predictions.', 'However, this uniqueness is not always the case, making it a critical area for further research.', 'The humanitarian impacts of conflict, depending on its type, scale, and many other factors, are extremely diffuse.', 'Meaningful anticipatory action on conflict requires not just the prediction of conflict events or risk, but also an ability to extend that prediction to the humanitarian impact of the conflict.', 'Understanding the relationship between conflict and humanitarian impact is critical for understanding how to set the parameters for any model.', 'Defining conflict onset or escalation or the type of conflict to predict relies on how linked these elements are to humanitarian impact.', 'To better link predictions of conflict to humanitarian impact, we should explore the use of more locally generated data.', 'We should also incorporate human judgment into the prediction models.', 'Alternative modeling techniques should be considered to make models more sensitive to dynamics on the ground.']}, {'header': 'PRACTICE TRANSPARENT DEVELOPMENT AND EVALUATION OF MODELS', 'text': ['Some methods have been implemented, but there is often a lack of clear and transparent testing and publication of model performance.', 'There is a lack of clear and transparent testing and publication of model performance, including through peer review processes.', 'All conflict prediction research, development, and implementation should use transparent and reproducible methods.', 'Open communication efforts are critical to building trust in complex models.', 'Open communication efforts are a prerequisite for achieving feasibility.', 'Transparency should not be restricted to the development of quantitative modeling.', 'All conflict prediction exercises, even human forecast-based methods, should strive to test and validate approaches.', 'Validation will allow end users to assign credibility to high performing models.', 'Validation will allow end users to recognize where further improvements are needed prior to implementation.', 'Ethical review of models and their applications is critical given the sensitivities of conflict-related programming.']}, {'header': 'CONCLUSION', 'text': ['Based on the review of the current literature, we return to the original questions framing our research.', 'The accuracy of conflict forecasts is being evaluated.', 'For classification models, the current evidence shows that model performance, particularly when predicting conflict onset or benchmarked against a simple model using conflict history, is not sufficient for application in humanitarian response.', 'There is a missing link between predicted conflict and humanitarian impact.', 'Risk and continuous prediction modeling requires additional evidence on how accurate risk estimates correspond to observed risk for responsible implementation and adoption.', 'The applications of risk and continuous prediction modeling in anticipatory action are less clear due to the lack of thresholds required to automate the release of funds.', 'The ability of various sources to predict different types of conflict in specific situations is being assessed.', 'For the purpose of this paper, we mainly focused on the overall performance of conflict prediction models rather than the predictive power of various sources.', 'Conflict history is the best predictor of future conflict but its utility is limited to ongoing conflicts.', 'Some sources can improve model performance, but the best performing models often predict conflict at a scale too small for meaningful application in humanitarian response.', 'These models do not significantly outperform simple models solely built on conflict history.', 'The current set of classification models in use or under development do not have sufficient predictive performance for anticipatory action.', 'The feasibility of application for risk and continuous models remains an open question that requires further research.']}, {'header': 'RECOMMENDATIONS AND NEXT STEPS', 'text': ['To make use of conflict prediction for anticipatory action in the humanitarian sector, focus on six areas.', 'Utilize flexible models that do not presuppose a theoretical framework of conflict causality.', 'Future work on developing conflict prediction models should build on the current literature.', 'The most fruitful areas of exploration will likely be found using ensemble models or non-linear techniques that require little theoretical knowledge and the use of as many input predictors as available.', 'Focus models on predicting shifts in conflicts, such as an increase in intensity or onset.', 'Explore the use of human inputs through superforecasters or prediction markets and use local data to improve model performance in specific contexts.', 'Work should be done to include and validate the usefulness of human inputs into model frameworks, such as through superforecasters or prediction markets.', 'Models can be validated and tested against their ability to predict the timing of specific events or conflict shifts.', 'Improve predictions on the humanitarian impact of conflict as opposed to conflict itself.', 'Work on predicting the humanitarian impact of conflict should be prioritized.', 'Understanding the relationship between humanitarian needs and conflict is necessary for any predictions to be acted on.', 'Even without predicting the onset or escalation of a conflict, improvements in predicting the humanitarian impacts of conflict could potentially enable anticipatory action on the impacts of observed conflict.', 'This work could be undertaken through better integration of conflict data and modeling with existing forecasting systems, which is already being explored for predicting food insecurity.', 'Ensure that model development and evaluation is done in a reproducible and transparent way that highlights the model performance in all relevant metrics.', 'Humanitarian work on measuring risk, predicting displacement, and other applications of predictive analytics often do not communicate validation statistics and evidence of good performance.', 'Recent efforts by governments and academics to predict conflict for use in humanitarian response also fail to show performance sufficient to what is required for use in decision making.', 'Regardless of the approach or methodology, it is critical that all models are transparent and that their performance is validated.', 'Learn from the state-of-the-art research underway in the academic field and ensure that applied research is relevant for humanitarian decision making.', 'Significant time and resources will be required to collect data, develop models, and test performance of new observations, particularly if models are extended to predict humanitarian impact.', 'It is only through engagement between conflict researchers, academia, the private sector, and the humanitarian community that this work will be sustainably financed and developed.', 'We hope that the above recommendations provide an initial framework and understanding for these conversations.', 'We welcome questions and feedback at centrehumdata@un.org.', '56 Joris J. L. Westerveld et al., 2021.']}, {'header': 'TYPES OF PREDICTION', 'text': ['The paper describes three model types: Classification, Risk prediction, and Continuous prediction.', 'Classification model predicts: There will be conflict.', 'Risk prediction model states: There is a 75 percent probability of conflict.', 'Continuous prediction model forecasts: There will be 33 fatalities due to conflict.', "The paper provides a detailed description and assessment of each model's usefulness and performance."]}, {'header': 'CLASSIFICATION', 'text': ['Classification models categorically state whether or not a conflict will occur in a particular area and time.', 'Classification models use binary (yes/no) or multiclass classification.', 'Binary models classify whether conflict will or will not occur in a single time period.', 'The classification may differ from the previous period.', 'Classification models are typically built on top of a probabilistic model.', 'Thresholds convert the probability of conflict into discrete classes.', 'A probability threshold is used to determine conflict occurrence.', 'For example, in Mali, conflict is defined as 25 or more conflict-related fatalities in one month.', 'A 50 percent probability threshold results in a classification forecast.', 'A 75 percent risk estimate predicts conflict in February 2022 with 25 or more fatalities.']}, {'header': 'RISK PREDICTION', 'text': ['Risk prediction is designed to generate a measure of underlying risk of conflict.', 'Risk prediction usually produces a probability of conflict at a certain geographic scale within a certain timeframe.', 'Probabilistic models underlie classification models in risk prediction.', 'Predictions in risk prediction are generated from a set of indicators.', 'Examples of indicators include gross domestic product and number of conflict events.', 'Probabilities in risk prediction typically range from near zero percent to close to 100 percent.', 'The ViEWS Risk Monitor and related conflict forecasting systems are examples of risk prediction models.', 'The ViEWS Risk Monitor publishes forecasts as risk predictions.', 'In the December 2021 ViEWS Risk Monitor, the risk of state-based violence resulting in 25 or more fatalities in Mali by February 2022 was estimated to be greater than 75 percent.']}, {'header': 'CONTINUOUS PREDICTION', 'text': ['Continuous prediction models predict a specific measure of conflict.', 'These measures include the number of fatalities or conflict events.', 'Continuous prediction models do not define conflict thresholds.', 'They are potentially better at capturing gradations in severity and magnitude.', 'Continuous prediction models may capture details missed in risk prediction or classification.', 'For Mali, the continuous prediction for February 2022 is an exact figure of 33 fatalities.']}, {'header': 'MODEL USEFULNESS', 'text': ['Classification outputs are often more useful to policy makers due to their ease of interpretation and actionability.', 'Classification outputs may not always be preferable to other models based on measures of predictive performance.', 'A model that generates probabilities of conflict uses a specific threshold for predictions.', 'Predictions greater than or equal to the threshold are predicted as being in conflict.', 'Predictions below the threshold are predicted as not being in conflict.', 'This requires decent separability of the data.', 'Decent separability means the threshold clearly delineates when conflict will occur versus when it will not.', 'If the data is well-separated, there is high confidence that conflict will occur if the threshold is met.', 'Risk prediction models can be used even if we do not have the confidence in predicting a specific event.', 'Intuitively, when a risk model predicts a 50 percent probability of conflict, the model performs well if conflict occurs approximately 50 percent of the time.', 'A threshold of 50 percent for classification would generate false positives 50 percent of the time.']}, {'header': 'MODEL PERFORMANCE', 'text': ['There are various metrics used to assess model performance.', 'The main difference in these metrics is between measuring calibration and discrimination.', 'Calibration refers to how well the probabilities of a model match observed frequencies.', 'Perfect calibration means that if a 50 percent probability of conflict is predicted, it occurs 50 percent of the time.', 'Poor calibration means the predicted probability does not accurately represent the actual likelihood of conflict.', 'Discrimination refers to how well the model separates different classifications using a specific threshold.', 'Perfect discrimination means conflict always occurs if the probability is above the threshold, and never if below.', 'Poor discrimination means conflict is equally likely regardless of the prediction above or below the threshold.', 'Models can be well-calibrated even with poor discrimination, and vice versa.']}, {'header': 'MODEL PERFORMANCE METRICS', 'text': ['Measures of calibration for classification and risk prediction models include: Brier score: Measures how close predicted probabilities of conflict are to the observed truth.', 'Continuous risk probability score (CRPS): Measures the accuracy of probabilistic forecasts against the observed distribution of outcomes.', 'Hosmer-Lemeshow test: Measures how closely the predicted probabilities match the observed occurrence of events.', 'Measures of discrimination for classification and risk prediction models include: Area Under the Receiver Operating Characteristic Curve (ROC AUC): The ability of the model to separate conflict events from non-conflict events at each potential probability threshold.', "Area Under the Precision-Recall Curve (AUPRC): Measure of the balance between precision and recall that does not use true negatives in its calculation. This makes it potentially important for conflict prediction to ensure that model evaluation is not solely driven by true predictions of ‘no conflict'.", 'Precision: The ability of the model to only predict true conflicts. Only applicable to classification models with set thresholds.', 'Recall: The ability of the model to predict all conflicts. Only applicable to classification models with set thresholds.', 'F1: Balanced measure of the model based on precision and recall. Only applicable to classification models with set thresholds.', 'Metrics for continuous prediction models include: Mean Squared Error (MSE): Common and simple measure of how close the predicted value is to the observed value.', 'Pseudo-Earth Mover Divergence (pEMDiv): A more complex measure that ensures that predictions that are only slightly off geographically or temporally are still slightly rewarded, rather than static accuracy measured for each prediction only against observations in that exact location and point in time.', 'Targeted Absolute Difference with Direction Augmentation (TADDA): Relevant for predictions of change but not point predictions. It is a measure of general accuracy that provides an additional penalty if the predicted direction of change is incorrect.']}, {'header': 'PERFORMANCE OF CLASSIFICATION MODELS', 'text': ['The performance of classification models is assessed by key metrics in the literature.', 'These metrics are calculated based on actual outcomes.', 'Actual outcomes involve comparing whether a conflict did or did not occur with the prediction.', 'Together, these metrics indicate a model’s usefulness.', 'They also suggest how a model should or should not be applied.']}, {'header': 'Advancements in model performance', 'text': ['Over the past two decades, academic research around conflict prediction has expanded.', 'Model performance in conflict prediction has improved.', 'Improvements in model performance are due to developments in global databases of conflict.', 'Databases such as ACLED, the UCDP events database, and CAMEO event data have improved access to data for model development.', 'New techniques for non-linear learning of complex data have been developed.', 'Techniques such as tree-based models, random forests, and neural networks have allowed more complex use of predictor variables.', 'These techniques do not presuppose a theoretical framework for conflict causality.', 'The result is more powerful and robust predictions.', 'Michael Ward et al. published results in 2013.', 'Recent years have seen increased predictive performance.', 'New iterations of the ViEWS models have been released.', 'Specific contextual models have been developed and benchmarked against ViEWS.', 'Automated machine learning models have outperformed these baselines.']}, {'header': 'Predictive power of conflict history', 'text': ['The advances in conflict prediction models have been driven by complex methodologies.']}, {'header': 'Efficacy of classification models in predicting conflict', 'text': ['Classification models lack the performance and granularity needed for anticipatory action.', 'Ongoing conflict drives high levels of humanitarian need.', 'Ongoing conflict dominates in predicting future conflict.', 'Responding to current humanitarian needs reaches populations likely to experience future conflict.', 'This approach differs from shocks like flooding, where future flooding cannot be anticipated by current flooding.', 'Classification models perform poorly in predicting new conflicts or intensifying ongoing conflicts.', 'There is little justification for using classification models for anticipatory action on conflict.', 'Improvements in conflict prediction are necessary for these methods to become operational.']}, {'header': 'PERFORMANCE OF RISK PREDICTION', 'text': ['Assessing the feasibility of risk prediction is a separate question from the feasibility of classification.', 'Risk prediction requires the predicted probability of conflict to closely match the observed frequency.', 'Calibration is described in the above methods section.']}, {'header': 'Lack of validation on calibration', 'text': ['Many classification models present their results as estimates of conflict risk.', 'Model validation and testing often use metrics designed to test discrimination between conflict occurrence and non-occurrence.', 'These error measurements do not inform the direct usefulness of the probability estimates.', 'Testing the calibration of the model is required to gauge how well predicted probabilities perform relative to observed distributions.', 'Proving the performance of risk models requires model validation on its calibration.', 'The conflict forecasting literature often focuses on predicting exactly when conflict will occur.', 'Measures of discrimination are frequently used, but measures of calibration are not explored in the same detail.', 'Limited plotting of metrics or additional measures are presented.', 'Having readily available evidence on how risk probability measures match empirically observed risk would allow end users to interpret and apply probabilities in planning humanitarian operations.', 'Risk estimates should perform well when predicting the risk of conflict onset, not just in existing conflict scenarios.', 'Applications of risk models should consider model limitations when extending predictions to humanitarian risk.', 'There may be a need for measuring the calibration of models when predicting large-scale escalation or onset.', 'Efficacy of risk models in predicting conflict often lacks sufficient exploration of model calibration.', 'Measures of risk estimate performance from other fields, such as the Hosmer-Lemeshow test and calibration plots, should be used.', 'Validating how well-calibrated risk prediction models are is a prerequisite for determining feasibility and connecting conflict risk to humanitarian impact risk.']}, {'header': 'PERFORMANCE OF CONTINUOUS PREDICTION', 'text': ['Continuous prediction models have historically been less common in the conflict prediction space.', 'Focus on continuous prediction models has increased in recent years.', 'The ViEWS research team organized an escalation prediction competition in 2020 and 2021.', 'The competition produced new research on continuous prediction specifically aimed at solving issues in conflict prediction.', 'The competition models focused on predicting significant changes in conflict, rather than point estimates.', 'The competition models were continuous prediction models.', "A suite of appropriate metrics was defined for the competition to reward different aspects of a model's performance.", 'These metrics would be useful for future research.', 'The source of this information is Håvard Hegre et al., 2022.']}, {'header': 'Burgeoning research but similar issues', 'text': ['The prediction competition has spurred on research.', 'This research aims to address many problems discussed above.', 'New models represent complex conflict dynamics and show increased performance in predicting conflict escalation.', 'Other models built on dynamic data, such as text data, perform better in predicting conflict outbreaks in peaceful settings.', 'The ViEWS competition summary paper details various advances and approaches.']}, {'header': 'Efficacy of continuous prediction models in predicting conflict', 'text': ['Substantive work has been put into developing metrics for assessing performance of continuous prediction models.', 'Risk prediction models differ from continuous prediction models in terms of the work put into developing performance metrics.', 'The scale of conflict predicted by continuous models is too small for likely application in anticipatory action.', 'Performance at the extremes of the distributions should be tested in continuous models to see if they can predict large-scale escalations in conflict.', 'Work on fatality prediction in continuous models has no direct linkage to humanitarian impact prediction.']}]

    docs = []

    with tqdm(
        total=len(organized_all_data),
        desc="Creating documents for vector store.",
        unit="Data",
    ) as pbar:
        for item in organized_all_data:
            header = (
                f"Header: {item['header'].capitalize()}\n"
                if item["header"] != "None"
                else ""
            )
            if "image" in item.keys():
                page_content = f"{header}Image: {item['image']}\nText: {item['text']}"
                docs.append(Document(page_content=page_content))

            elif "table" in item.keys():
                page_content = f"{header}Table: {item['table']}\nText: {item['text']}"
                docs.append(Document(page_content=page_content))

            else:
                for text in item["text"]:
                    page_content = f"{header}Text: {text}"
                    docs.append(Document(page_content=page_content))

            pbar.update(1)

    print(f"Created {len(docs)} documents for Chroma vector store.")

    if os.path.exists(path):
        shutil.rmtree(path)

    vector_store = Chroma.from_documents(
        documents=docs,
        collection_name="collection",
        embedding=embedding_model,
        persist_directory=path,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Saved documents in Chroma vector store.")


if __name__ == "__main__":

    # start = perf_counter()
    #
    # save_vector_to_store(
    #     path="../vectordb_chroma/",
    #     data_path="../../data/Pdf/2a85b52768ea5761b773be49b09d15f0b95415b0.pdf",
    # )
    #
    # print(f"Total execution time: {perf_counter() - start:.2f} seconds.")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name="collection",
        persist_directory="../vectordb_chroma/",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 100}
    )

    query = "who reviewed the study?"

    retrieved_docs = retriever.get_relevant_documents(query)
    print(retrieved_docs)
    print(len(retrieved_docs))

    chroma_results = vector_store.similarity_search_with_relevance_scores(query, k=100)
    print(chroma_results)
    print(len(chroma_results))

