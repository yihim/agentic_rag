# Model names
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
QWEN = "Qwen/Qwen2.5-14B-Instruct"

# vLLM
VLLM_BASE_URL = "http://localhost:8080/v1"
VLLM_API_CHAT_COMPLETIONS_URL = F"{VLLM_BASE_URL}/chat/completions"
VLLM_API_REQUEST_PAYLOAD_TEMPLATE = {
    "model": "/model",
    "seed": 42,
    "temperature": 0.01,
    "top_p": 0.8,
    "repetition_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "top_k": 20,
}

# Max Tokens Length
TABLE_ORGANIZER_LLM_MAX_TOKENS = 2560
AGENTIC_CHUNKER_LLM_MAX_TOKENS = 2560

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
- Do not include the JSON schema
- Focus on meaningful, contextual description
- Be precise and informative

Output Example:
{
  "description": "Precise narrative explaining the table's content and significance"
}
"""

AGENTIC_CHUNKER_LLM_SYSTEM_PROMPT = """
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
    1. Express a Single Fact: Each proposition should state one specific fact or claim.
    2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
    3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
    4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
    5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.
    6. Present the results as a JSON formatted list of strings only.

Example:

Input:
Decompose the following:  
Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares  
Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring.  

There are several factors that contributed to the popularity of the Easter Hare:  
- The influence of Easter cards  
- The presence of toys depicting the Easter Hare  
- Books that popularized the tradition  

In the nineteenth century, these factors made the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America, where it evolved into the Easter Bunny.  

Output: 
["The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter.", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century, the popularity of the Easter Hare/Rabbit was influenced by Easter cards, toys, and books.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America."]
"""

AGENT_ROUTER_SYSTEM_PROMPT = """

"""

QUERY_REWRITER_SYSTEM_PROMPT = """

"""

RESPONSE_CHECKER_SYSTEM_PROMPT = """

"""
