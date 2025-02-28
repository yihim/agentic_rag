import re
import string


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
        json_output.append({"header": "None", "content": "\n".join(non_header_content)})

    # Convert to JSON with ensure_ascii=False to preserve Unicode characters
    return json_output


def clean_references(text):
    # This pattern specifically targets academic citation patterns
    # Example: "conflict.24, 25, 26" or "performance.27"
    citation_pattern = r"([a-zA-Z])\.\s*(\d+(?:\s*,\s*\d+)*)"

    # Replace with just the word followed by a period
    cleaned_text = re.sub(citation_pattern, r"\1.", text)

    return cleaned_text
