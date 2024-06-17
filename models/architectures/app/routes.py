from flask import Blueprint, request, jsonify, render_template
from .finetune import finetune_model
from .verify import verify_output
import logging
log = logging.getLogger(__name__)
main = Blueprint('main', __name__)


@main.route('/')
def index():
    log.info('Index page accessed')
    return render_template('index.html')

@main.route('/finetune_page')
def finetune_page():
    return render_template('finetune.html')


@main.route('/verify_page')
def verify_page():
    return render_template('verify.html')

@main.route('/finetune', methods=['POST'])
def finetune():
    data = request.json
    result = finetune_model(data)
    return jsonify(result)

@main.route('/verify', methods=['POST'])
def verify():
    data = request.json
    result = verify_output(data)
    return jsonify(result)
