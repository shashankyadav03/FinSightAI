import requests

# Your GitHub personal access token
token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# The repository where you want to create issues
repo = 'shashankyadav03/FinSightAI'
# The GitHub API URL for creating an issue
url = f'https://api.github.com/repos/{repo}/issues'

headers = {
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json',
}

# List of your tasks
tasks = [
    "Logging: Implement logging_utils.py for project-wide logging.",
    "Configuration: Set up config_parser.py for managing project configurations.",
    "Model Serialization: Develop model_saver_loader.py for saving and loading models.",
    "Infrastructure and Deployment",
    "Dockerize Application: Containerize the FinGPT model for easy deployment.",
    "CI/CD Pipeline: Set up continuous integration and deployment workflows.",
    "Documentation and Collaboration",
    "README Update: Comprehensive project documentation.",
    "Contribution Guidelines: Define how others can contribute to the project.",
    "Code of Conduct: Establish a code of conduct for project contributors."
]

def create_issue(title):
    """Create an issue on GitHub."""
    data = {'title': title}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f'Successfully created issue "{title}"')
    else:
        print(f'Failed to create issue "{title}": {response.content}')

for task in tasks:
    create_issue(task)
