# Model names
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
TABLE_ORGANIZER_LLM = "tiiuae/Falcon3-10B-Instruct"
VISION_LLM = "meta-llama/Llama-3.2-11B-Vision-Instruct"
AGENTIC_CHUNKER_LLM = "tiiuae/Falcon3-10B-Instruct"

# Max Tokens Length
TABLE_ORGANIZER_LLM_MAX_TOKENS = 1024
VISION_LLM_MAX_TOKENS = 1024
AGENTIC_CHUNKER_LLM_MAX_TOKENS = 1024

# Prompt templates
TABLE_ORGANIZER_LLM_SYSTEM_PROMPT = """
You are a HTML table analysis expert. Your core objectives are:

1. Create a concise, insightful contextual description of the table
2. Only provide your output with a precise JSON object capturing the table's essence

Key Analysis Dimensions:
- Identify the table's primary purpose
- Highlight key data patterns or insights
- Extract critical information succinctly

Output Requirements:
- Strictly use the JSON format
- Focus on meaningful, contextual description
- Be precise and informative

Output Example:
{
  "description": "Precise narrative explaining the table's content and significance"
}
"""

VISION_LLM_SYSTEM_PROMPT = """
You are a vision-language model tasked with analyzing images. 
Your role is to provide concise, clear, and accurate descriptions of the content within an image. 
Focus on identifying the primary subject, key details, and overall context, while avoiding unnecessary speculation or overly detailed observations. 
Be objective and neutral in your descriptions.
Provide your response in one paragraph.
"""

AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT = """
Please decompose the following content into simple, self-contained propositions. Ensure that each proposition meets the following criteria:
    1. Express a Single Fact: Each proposition should state one specific fact or claim.
    2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
    3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
    4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
    5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.
    6. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    7. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
    8. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
    9. Present the results as a list of strings, formatted in JSON.

Example:

Input:
Decompose the following:
Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares
Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny.

Output: 
["The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America."]
"""