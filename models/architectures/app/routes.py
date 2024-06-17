from flask import Blueprint, request, jsonify, render_template
from .finetune import fine_tune
from .verify import verify
from .chat import chat_with_model
import logging

log = logging.getLogger(__name__)
main = Blueprint('main', __name__)

@main.route('/')
def index():
    log.info('Index page accessed')
    return render_template('index.html')

@main.route('/finetune', methods=['GET', 'POST'])
def finetune_route():
    if request.method == 'POST':
        return fine_tune()
    return render_template('finetune.html')

@main.route('/verify', methods=['GET', 'POST'])
def verify_route():
    if request.method == 'POST':
        return verify()
    output = request.args.get('output', '')
    return render_template('verify.html', output=output)

@main.route('/chat', methods=['GET', 'POST'])
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
