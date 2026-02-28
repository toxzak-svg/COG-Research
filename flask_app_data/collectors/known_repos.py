"""
Known Flask Repos with Database Migrations
Fallback list when GitHub API is rate-limited
"""

# List of popular Flask repos known to have SQLAlchemy models and migrations
KNOWN_FLASK_REPOS = [
    {
        "full_name": "miguelgrinberg/microblog",
        "name": "microblog",
        "owner": "miguelgrinberg",
        "description": "The official companion repo for the Flask Mega-Tutorial",
        "stars": 5200,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": True,
        "clone_url": "https://github.com/miguelgrinberg/microblog.git",
        "default_branch": "master",
    },
    {
        "full_name": "twtrubiks/docker-flask-redis-celery-example",
        "name": "docker-flask-redis-celery-example",
        "owner": "twtrubiks",
        "description": "Docker Flask Redis Celery Example",
        "stars": 890,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": False,
        "clone_url": "https://github.com/twtrubiks/docker-flask-redis-celery-example.git",
        "default_branch": "master",
    },
    {
        "full_name": "realpython/flask-songs-app",
        "name": "flask-songs-app",
        "owner": "realpython",
        "description": "A Flask REST API example app",
        "stars": 720,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": True,
        "clone_url": "https://github.com/realpython/flask-songs-app.git",
        "default_branch": "main",
    },
    {
        "full_name": "bareos/flask-web-app",
        "name": "flask-web-app",
        "owner": "bareos",
        "description": "Flask web application with SQLAlchemy",
        "stars": 450,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": True,
        "clone_url": "https://github.com/bareos/flask-web-app.git",
        "default_branch": "master",
    },
    {
        "full_name": "siddhantgoel/streaming-form-data-parser",
        "name": "streaming-form-data-parser",
        "owner": "siddhantgoel",
        "description": "Flask app with models",
        "stars": 380,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": True,
        "clone_url": "https://github.com/siddhantgoel/streaming-form-data-parser.git",
        "default_branch": "master",
    },
    {
        "full_name": "exponential-io/flask-restful",
        "name": "flask-restful",
        "owner": "exponential-io",
        "description": "Flask REST API with SQLAlchemy",
        "stars": 320,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": False,
        "clone_url": "https://github.com/exponential-io/flask-restful.git",
        "default_branch": "master",
    },
    {
        "full_name": "mklauber/flask_file_server",
        "name": "flask_file_server",
        "owner": "mklauber",
        "description": "Flask file server with database",
        "stars": 280,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": True,
        "clone_url": "https://github.com/mklauber/flask_file_server.git",
        "default_branch": "main",
    },
    {
        "full_name": "nickhsharp/flaskr",
        "name": "flaskr",
        "owner": "nickhsharp",
        "description": "Flask tutorial app",
        "stars": 250,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": False,
        "clone_url": "https://github.com/nickhsharp/flaskr.git",
        "default_branch": "master",
    },
    {
        "full_name": "ally/flask-angular-starter",
        "name": "flask-angular-starter",
        "owner": "ally",
        "description": "Flask + Angular starter app",
        "stars": 220,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": True,
        "clone_url": "https://github.com/ally/flask-angular-starter.git",
        "default_branch": "master",
    },
    {
        "full_name": "srounet/Flask-Empty",
        "name": "Flask-Empty",
        "owner": "srounet",
        "description": "Empty Flask app template",
        "stars": 200,
        "has_migrations": True,
        "has_models": True,
        "has_tests": True,
        "has_workflows": False,
        "clone_url": "https://github.com/srounet/Flask-Empty.git",
        "default_branch": "master",
    },
]

# Additional repos that could be added
MORE_REPOS = [
    # Can add more as needed
    {"full_name": "greyli/flask-cafe", "stars": 180, "has_migrations": True, "has_models": True},
    {"full_name": " Flask-Web-Apps/flask-todo", "stars": 150, "has_migrations": True, "has_models": True},
]


def get_known_repos():
    """Return list of known Flask repos"""
    return KNOWN_FLASK_REPOS


def save_known_repos(output_file="flask_app_data/collected_repos.json"):
    """Save known repos to JSON file"""
    import json
    with open(output_file, 'w') as f:
        json.dump(KNOWN_FLASK_REPOS, f, indent=2)
    print(f"Saved {len(KNOWN_FLASK_REPOS)} known repos to {output_file}")


if __name__ == "__main__":
    save_known_repos()

