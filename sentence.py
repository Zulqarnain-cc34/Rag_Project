from sentence_transformers import SentenceTransformer
import torch
import time

# Enable TF32 for better performance
torch.set_float32_matmul_precision("high")

model_name = "nomic-ai/modernbert-embed-base"
model = SentenceTransformer(model_name)


def get_embeddings(model, sentences):
    # Add search_query prefix to each sentence as shown in the example
    prefixed_sentences = [f"search_query: {sentence}" for sentence in sentences]
    return model.encode(prefixed_sentences)


def get_similarity(model, embedding1, embedding2):
    return model.similarity(embedding1, embedding2)


def main():
    print("# Semantic Search Analysis Report")
    print("\n## Model Information")
    print(f"- Using embedding model: `{model_name}`")

    requirements = [
        "Used React on a project",
        "Expert level at Angular",
        "Years of Python experience",
        "Worked with Large Language Models",
        "I am looking for an AI guru",
        "Has to be a senior developer",
        "Has a Computer Science Degree",
        "Knows JavaScript and TypeScript",
        "Leads dev teams",
        "Contributes to open source",
    ]

    propositions = [
        "Candidate knows Generative AI.",
        "Candidate has experience with Generative AI.",
        "Candidate has worked with Generative AI technologies.",
        "Candidate knows OpenAI.",
        "Candidate has experience with OpenAI.",
        "Candidate has worked with OpenAI tools.",
        "Candidate knows Open Source Models.",
        "Candidate has experience with Open Source Models.",
        "Candidate has worked with Open Source Models.",
        "Candidate knows Fine-tuning.",
        "Candidate has experience with Fine-tuning.",
        "Candidate has worked with Fine-tuning techniques.",
        "Candidate knows Function Calling.",
        "Candidate has experience with Function Calling.",
        "Candidate has worked with Function Calling in AI.",
        "Candidate knows Agents.",
        "Candidate has experience with Agents.",
        "Candidate has worked with AI Agents.",
        "Candidate knows ChatBots.",
        "Candidate has experience with ChatBots.",
        "Candidate has worked with ChatBots development.",
        "Candidate knows React.",
        "Candidate has experience with React.",
        "Candidate has worked with React framework.",
        "Candidate knows TypeScript.",
        "Candidate has experience with TypeScript.",
        "Candidate has worked with TypeScript programming language.",
        "Candidate knows Node.",
        "Candidate has experience with Node.js.",
        "Candidate has worked with Node.js environment.",
        "Candidate knows Bun.",
        "Candidate has experience with Bun.",
        "Candidate has worked with Bun package manager.",
        "Candidate knows Python FastAPI.",
        "Candidate has experience with Python FastAPI.",
        "Candidate has worked with Python FastAPI framework.",
        "Candidate knows GraphQL.",
        "Candidate has experience with GraphQL.",
        "Candidate has worked with GraphQL API.",
        "Candidate knows PostgreSQL.",
        "Candidate has experience with PostgreSQL.",
        "Candidate has worked with PostgreSQL database.",
        "Candidate knows Vector Databases.",
        "Candidate has experience with Vector Databases.",
        "Candidate has worked with Vector Databases technology.",
        "Candidate knows Docker.",
        "Candidate has experience with Docker.",
        "Candidate has worked with Docker containerization.",
        "Candidate knows Linux.",
        "Candidate has experience with Linux.",
        "Candidate has worked with Linux operating system.",
        "Candidate knows CI/CD.",
        "Candidate has experience with CI/CD.",
        "Candidate has worked with CI/CD pipelines.",
        "Candidate knows GitHub Actions.",
        "Candidate has experience with GitHub Actions.",
        "Candidate has worked with GitHub Actions for CI/CD.",
        "Candidate knows GitLab Pipeline.",
        "Candidate has experience with GitLab Pipeline.",
        "Candidate has worked with GitLab Pipeline for CI/CD.",
        "Candidate knows Azure DevOps.",
        "Candidate has experience with Azure DevOps.",
        "Candidate has worked with Azure DevOps for CI/CD.",
        "Candidate has Tech Leadership experience.",
        "Candidate is a Tech Leader.",
        "Candidate leads technical teams.",
        "Candidate mentors junior engineers.",
        "Candidate trains junior engineers.",
        "Candidate has experience mentoring junior engineers.",
        "Candidate interacts with clients.",
        "Candidate has client interaction experience.",
        "Candidate communicates with clients effectively.",
        "Candidate uses Scrum.",
        "Candidate practices Agile.",
        "Candidate has experience with Scrum and Agile methodologies.",
        "Candidate creates AI policy.",
        "Candidate develops organizational AI policy.",
        "Candidate has experience creating organizational-wide AI policy.",
        "Candidate is a Principal AI Engineer at Umbrage.",
        "Candidate works as a Principal AI Engineer at Umbrage.",
        "Candidate has been a Principal AI Engineer at Umbrage since December 2019.",
        "Candidate is a Director of Engineering at Umbrage.",
        "Candidate leads Engineering at Umbrage.",
        "Candidate has led Engineering at Umbrage as a Director.",
        "Candidate is a Senior Web Engineer at Umbrage.",
        "Candidate has senior experience in Web Engineering at Umbrage.",
        "Candidate has been a Senior Web Engineer at Umbrage.",
        "Candidate is a Web Developer at Accenture.",
        "Candidate works as a Web Developer at Accenture.",
        "Candidate has been a Web Developer at Accenture since June 2018.",
        "Candidate is a Self Employed App Developer.",
        "Candidate works as a Freelance App Developer.",
        "Candidate has been a Self Employed App Developer since March 2013.",
        "Candidate has a Professional Certification in Graph Development from Apollo GraphQL.",
        "Candidate is certified in Graph Development from Apollo GraphQL.",
        "Candidate earned a Professional Certification in Graph Development from Apollo GraphQL in January 2023.",
        "Candidate has an Associate Certification in Graph Development from Apollo GraphQL.",
        "Candidate is certified at the Associate level in Graph Development from Apollo GraphQL.",
        "Candidate earned an Associate Certification in Graph Development from Apollo GraphQL in May 2022.",
        "Candidate completed a Coding Bootcamp in Web Development at UCSD Extension.",
        "Candidate has a Coding Bootcamp background in Web Development from UCSD Extension.",
        "Candidate attended a Coding Bootcamp in Web Development at UCSD Extension from 2017 to 2018.",
        "Candidate has a certification in UNIX and Linux Systems Administration from UCSD Extension.",
        "Candidate is certified in UNIX and Linux Systems Administration from UCSD Extension.",
        "Candidate earned a UNIX and Linux Systems Administration Certification from UCSD Extension from 2013 to 2014.",
    ]

    print("\n## Requirements")
    for i, req in enumerate(requirements, 1):
        print(f"{i}. {req}")

    print("\n## Resume Propositions")
    for i, proposition in enumerate(propositions, 1):
        print(f"{i}. {proposition}")

    # Get embeddings for requirements and propositions
    print("\n## Processing")
    print("Generating embeddings...")
    req_embeddings = get_embeddings(model, requirements)
    proposition_embeddings = get_embeddings(model, propositions)

    # Calculate similarities between all pairs
    print("Calculating similarities...")
    start_time = time.time()
    similarities = []
    for i, req in enumerate(requirements):
        for j, proposition in enumerate(propositions):
            similarity = get_similarity(
                model, req_embeddings[i], proposition_embeddings[j]
            )

            # Convert tensor to float if needed
            if isinstance(similarity, torch.Tensor):
                similarity = similarity.item()

            similarities.append(
                {
                    "Requirement": req,
                    "Proposition": proposition,
                    "Similarity": similarity,
                }
            )

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x["Similarity"], reverse=True)

    # Create a dictionary to store highest similarity for each requirement
    highest_similarities = {}
    for sim in similarities:
        req = sim["Requirement"]
        if (
            req not in highest_similarities
            or sim["Similarity"] > highest_similarities[req]["Similarity"]
        ):
            highest_similarities[req] = sim

    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000

    # Print highest similarity for each requirement
    print("\n## Best Matches")
    total_similarity = 0
    for req, match in highest_similarities.items():
        similarity_percentage = match["Similarity"] * 100
        total_similarity += match["Similarity"]
        print(f"\n### Requirement: {req}")
        print(f"**Best matching proposition:** {match['Proposition']}")
        print(f"**Similarity:** {similarity_percentage:.2f}%")

    # Calculate and print AI Relevancy Score
    average_cosine_similarity = (total_similarity / len(requirements)) * 100

    # Calculate Win/Loss Ratio Score
    wins = sum(1 for sim in highest_similarities.values() if sim["Similarity"] >= 0.5)
    losses = len(requirements) - wins
    win_loss_score = (wins / len(requirements)) * 100 if len(requirements) > 0 else 0

    print("\n## Summary")
    print(f"- **Average Cosine Similarity:** {average_cosine_similarity:.2f}%")
    print(
        f"- **Requirements Met Score:** {win_loss_score:.2f}% ({wins} met / {losses} not met)"
    )
    print(f"- **Processing Time:** {processing_time_ms:.2f}ms")
