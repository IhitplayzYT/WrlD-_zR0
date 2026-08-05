import base64
import requests
import consts



def describe_image(image: bytes) -> str:
    image_b64 = base64.b64encode(image).decode()
    prompt = """
You are converting a PDF into Markdown.
Analyze this image and produce a Markdown deep description.
Do not try to infer meanings of charts/graph simply give numerically accurate data presened in the diagram
This will be feed to a RL Finance Model so do not add any hallucinated/fake data or inferences


Rules:
- If it is a chart, explain axes, legend, trends and important values.
- If it is a table, recreate it as a Markdown table.
- If it is a diagram, explain every node and relationship.
- If it contains equations, output them in LaTeX.
- If it is a screenshot, transcribe all visible text.
- If it is decorative or irrelevant, say:
  "Irrelevant image."

Return ONLY Markdown.
"""

    r = requests.post(
        consts.OLLAMA_URL,
        json={
            "model": consts.MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"].strip()