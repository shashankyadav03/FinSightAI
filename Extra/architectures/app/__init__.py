from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config['SECRET_KEY'] = 'my_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://finuser:finpass@localhost:5432/finsightdb'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    csrf.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = 'main.login'  # Specify the login view

    from .routes.routes import main
    with app.app_context():
        from .models.models import Users  # Import Users model inside app context
        db.create_all()  # Ensure that tables are created if they don't exist

    app.register_blueprint(main)

    return app

@login_manager.user_loader
def load_user(user_id):
    from .models.models import Users
    try:
        return Users.query.get(int(user_id))
    except ValueError:
        return None  # Return None if user_id is not an integer
