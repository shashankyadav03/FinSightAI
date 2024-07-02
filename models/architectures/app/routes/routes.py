from flask import Blueprint, request, jsonify, render_template
from ..services.finetune import fine_tune
from ..services.verify import verify
from ..services.chat import chat_with_model
import logging
from flask_login import LoginManager, login_user, UserMixin, login_required, logout_user
from ..models.models import Users
from ..models.forms import RegistrationForm, LoginForm
from .. import db

log = logging.getLogger(__name__)
main = Blueprint('main', __name__)

# Create an instance of the LoginManager
login_manager = LoginManager()
login_manager.init_app(main)
login_manager.login_view = 'main.login'  # Specify the login view

@login_manager.user_loader
def load_user(user_id):
    try:
        return Users.query.get(int(user_id))
    except ValueError:
        return None  # Return None if user_id is not an integer

@main.route('/')
def index():
    log.info('Index page accessed')
    return render_template('index.html')

# Define register route
@main.route('/register', methods=['POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user_data = form.data

        # Check if user already exists
        existing_user = Users.query.filter_by(email=user_data['email']).first()
        if existing_user:
            return jsonify({"error": "User already exists."}), 400

        # Create new user
        new_user = Users(username=user_data['username'], email=user_data['email'])
        new_user.set_password(user_data['password'])
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "User created successfully."}), 201
    else:
        return jsonify({"errors": form.errors}), 400

@main.route('/login', methods=['POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user_data = form.data

        # Check if user exists
        user = Users.query.filter_by(email=user_data['email']).first()
        if not user or not user.check_password(user_data['password']):
            return jsonify({"error": "Invalid email or password."}), 400

        # Log user in
        login_user(user)
        return jsonify({"message": "User logged in successfully."}), 200
    else:
        return jsonify({"errors": form.errors}), 400


    
@main.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({"message": "User logged out successfully."}), 200

@main.route('/finetune', methods=['GET', 'POST'])
@login_required
def finetune_route():
    if request.method == 'POST':
        return fine_tune()
    return render_template('finetune.html')

@main.route('/verify', methods=['GET', 'POST'])
@login_required
def verify_route():
    if request.method == 'POST':
        return verify()
    output = request.args.get('output', '')
    return render_template('verify.html', output=output)

@main.route('/chat', methods=['GET', 'POST'])
@login_required
def chat_route():
    if request.method == 'POST':
        return chat_with_model()
    return render_template('chat.html')

@main.app_errorhandler(404)
def page_not_found(e):
    log.error('Page not found: %s', (e))
    return jsonify({"error": "Page not found"}), 404

@main.app_errorhandler(500)
def internal_server_error(e):
    log.error('Server error: %s', (e))
    return jsonify({"error": "Internal server error"}), 500
