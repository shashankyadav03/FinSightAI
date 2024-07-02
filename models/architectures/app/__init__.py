from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

def create_app():
    app = Flask(__name__)
    CORS(app)

    db = SQLAlchemy(app)

    from .routes.routes import main
    app.register_blueprint(main)

    return app
