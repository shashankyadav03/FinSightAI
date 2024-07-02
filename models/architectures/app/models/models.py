from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from .. import db



class Users(db.Model):
    userid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    passwordhash = db.Column(db.String(255), nullable=False)
    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.userid)

    def set_password(self, password):
        self.passwordhash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.passwordhash, password)
    


