import base64
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyotp
from cryptography.fernet import Fernet, InvalidToken
from flask import request
from sqlalchemy import inspect, text, or_
from werkzeug.security import generate_password_hash


app = None
db = None
User = Camera = Settings = OtpChallenge = EventLog = RecordingRequest = None
BASE_RECORDINGS_DIR = 'users_data'
AUDIT_EVENT_TYPES = {
    'login', 'logout', 'create', 'delete', 'settings', 'update',
    'approve', 'deny', 'fulfill', 'add', 'remove', 'clear', 'reset',
    'factory_reset', 'export', 'archive', 'restore',
}


def configure(application, database, models, recordings_dir='users_data'):
    global app, db, User, Camera, Settings, OtpChallenge, EventLog, RecordingRequest, BASE_RECORDINGS_DIR
    app = application
    db = database
    User, Camera, Settings, OtpChallenge, EventLog, RecordingRequest = models
    BASE_RECORDINGS_DIR = recordings_dir


def get_user_role(user):
    if user is None:
        return 'user'
    return str(getattr(user, 'role', None) or 'user').strip().lower() or 'user'


def is_admin_user(user_or_email):
    if user_or_email is None:
        return False
    admin_emails = [email.strip().lower() for email in os.environ.get('ADMIN_EMAILS', '').split(',') if email.strip()]
    if isinstance(user_or_email, str):
        email = user_or_email.strip().lower()
        if email in admin_emails:
            return True
        user = User.query.filter_by(email=email).first()
        return bool(user and user.archived_at is None and get_user_role(user) == 'admin')
    email = getattr(user_or_email, 'email', '').strip().lower()
    return (
        getattr(user_or_email, 'archived_at', None) is None
        and (email in admin_emails or get_user_role(user_or_email) == 'admin')
    )


def get_client_ip():
    forwarded_for = request.headers.get('X-Forwarded-For') if request else None
    if forwarded_for:
        return forwarded_for.split(',')[0].strip() or request.remote_addr
    return request.remote_addr if request else None


def record_audit_event(user_id, event_type, description, source_name='System', ip_address=None):
    if user_id is None:
        return None
    normalized_type = (event_type or 'settings').strip().lower()
    if normalized_type not in AUDIT_EVENT_TYPES:
        normalized_type = 'settings'
    safe_description = (description or '').strip() or f'{normalized_type.title()} action recorded'
    try:
        with app.app_context():
            entry = EventLog(
                user_id=user_id,
                source_name=source_name or 'System',
                event_type=normalized_type,
                event_category='audit',
                description=safe_description[:500],
                ip_address=ip_address or get_client_ip(),
                actor_name=getattr(db.session.get(User, user_id), 'username', None),
                actor_email=getattr(db.session.get(User, user_id), 'email', None),
            )
            db.session.add(entry)
            db.session.commit()
            return entry
    except Exception:
        db.session.rollback()
        return None


def get_dashboard_target(user):
    return 'admin_dashboard' if is_admin_user(user) else 'index'


def _security_fernet():
    secret = app.config['SECRET_KEY']
    if isinstance(secret, str):
        secret = secret.encode()
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret).digest()))


def encrypt_totp_secret(secret):
    return _security_fernet().encrypt(secret.encode()).decode()


def decrypt_totp_secret(encrypted_secret):
    if not encrypted_secret:
        return None
    try:
        return _security_fernet().decrypt(encrypted_secret.encode()).decode()
    except (InvalidToken, ValueError):
        return None


def get_or_create_totp_secret(user):
    secret = decrypt_totp_secret(user.totp_secret)
    if secret:
        return secret
    secret = pyotp.random_base32()
    user.totp_secret = encrypt_totp_secret(secret)
    db.session.commit()
    return secret


def send_transactional_email(recipient, subject, body, html_body=None, attachments=None):
    api_key = app.config.get('RESEND_API_KEY')
    sender = app.config.get('RESEND_FROM_EMAIL')
    if not api_key or not sender or not recipient:
        app.logger.warning('Email delivery could not be attempted because Resend configuration or recipient is missing.')
        return False

    payload = {
        'from': sender,
        'to': [recipient],
        'subject': subject,
        'text': body or '',
    }
    if html_body:
        payload['html'] = html_body
    if attachments:
        payload['attachments'] = attachments

    request = Request(
        'https://api.resend.com/emails',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urlopen(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                app.logger.warning('Resend email delivery returned HTTP %s.', response.status)
                return False
        return True
    except HTTPError as exc:
        error_body = exc.read(2048).decode('utf-8', errors='replace')
        app.logger.warning('Resend email delivery failed with HTTP %s: %s', exc.code, error_body)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        app.logger.warning('Resend email delivery failed: %s', exc)
    return False


def send_security_email(user, subject, body, html_body=None):
    if not user or not user.email:
        app.logger.warning('Security email could not be sent because the recipient is missing.')
        return False
    return send_transactional_email(user.email, subject, body, html_body)


def create_email_otp(user):
    OtpChallenge.query.filter_by(user_id=user.id, purpose='password_reset', used_at=None).update({'used_at': datetime.utcnow()})
    code = f'{secrets.randbelow(1000000):06d}'
    db.session.add(OtpChallenge(user_id=user.id, purpose='password_reset', code_hash=generate_password_hash(code),
                                expires_at=datetime.utcnow() + timedelta(minutes=3)))
    db.session.commit()
    return code


def create_password_reset_token(user):
    OtpChallenge.query.filter_by(
        user_id=user.id, purpose='password_reset_link', used_at=None
    ).update({'used_at': datetime.utcnow()})
    token = secrets.token_urlsafe(32)
    db.session.add(OtpChallenge(
        user_id=user.id,
        purpose='password_reset_link',
        code_hash=hashlib.sha256(token.encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(hours=1),
    ))
    db.session.commit()
    return token


def create_login_email_otp(user):
    """Invalidate any open login OTP challenges and issue a fresh 6-digit code."""
    OtpChallenge.query.filter_by(user_id=user.id, purpose='login', used_at=None).update({'used_at': datetime.utcnow()})
    code = f'{secrets.randbelow(1000000):06d}'
    db.session.add(OtpChallenge(user_id=user.id, purpose='login', code_hash=generate_password_hash(code),
                                expires_at=datetime.utcnow() + timedelta(minutes=3)))
    db.session.commit()
    return code


def clear_vision_events_for_user(user_id=None):
    """Remove computer-vision events without ever touching the audit trail."""
    query = EventLog.query
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    return query.filter(~EventLog.event_type.in_(list(AUDIT_EVENT_TYPES))).delete(synchronize_session=False)


def ensure_database_schema():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        user_columns = {column['name'] for column in inspector.get_columns('user')}
        for name, definition in [
            ('role', "VARCHAR(20) NOT NULL DEFAULT 'user'"),
            ('is_approved', 'BOOLEAN NOT NULL DEFAULT 1'),
            ('totp_secret', 'VARCHAR(500)'),
            ('totp_enabled', 'BOOLEAN NOT NULL DEFAULT 0'),
            ('archived_at', 'DATETIME'),
            ('rejected_at', 'DATETIME'),
            ('building_number', 'VARCHAR(150)'),
            ('floor', 'VARCHAR(150)'),
            ('street_address', 'VARCHAR(200)'),
        ]:
            if name not in user_columns:
                db.session.execute(text(f'ALTER TABLE user ADD COLUMN {name} {definition}'))
                db.session.commit()
        camera_columns = {column['name'] for column in inspector.get_columns('camera')}
        if 'zone' not in camera_columns:
            db.session.execute(text('ALTER TABLE camera ADD COLUMN zone VARCHAR(100) NOT NULL DEFAULT "Main Entrance"'))
            db.session.commit()
        if 'is_public' not in camera_columns:
            db.session.execute(text('ALTER TABLE camera ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 0'))
            db.session.commit()
        if 'virtual_lines' not in camera_columns:
            db.session.execute(text('ALTER TABLE camera ADD COLUMN virtual_lines TEXT NOT NULL DEFAULT "[]"'))
            db.session.commit()
        settings_columns = {column['name'] for column in inspector.get_columns('settings')}
        if 'critical_email_cooldown_minutes' not in settings_columns:
            db.session.execute(text('ALTER TABLE settings ADD COLUMN critical_email_cooldown_minutes INTEGER NOT NULL DEFAULT 15'))
            db.session.commit()
        if 'last_critical_email_at' not in settings_columns:
            db.session.execute(text('ALTER TABLE settings ADD COLUMN last_critical_email_at DATETIME'))
            db.session.commit()
        event_log_columns = {column['name'] for column in inspector.get_columns('event_log')}
        if 'camera_id' not in event_log_columns:
            db.session.execute(text('ALTER TABLE event_log ADD COLUMN camera_id INTEGER'))
            db.session.commit()
        if 'confidence' not in event_log_columns:
            db.session.execute(text('ALTER TABLE event_log ADD COLUMN confidence FLOAT'))
            db.session.commit()
        if 'actor_name' not in event_log_columns:
            db.session.execute(text('ALTER TABLE event_log ADD COLUMN actor_name VARCHAR(150)'))
            db.session.commit()
        if 'actor_email' not in event_log_columns:
            db.session.execute(text('ALTER TABLE event_log ADD COLUMN actor_email VARCHAR(150)'))
            db.session.commit()
        if 'snapshot_path' not in event_log_columns:
            db.session.execute(text('ALTER TABLE event_log ADD COLUMN snapshot_path VARCHAR(255)'))
            db.session.commit()
        event_log_user_column = next(
            column for column in inspector.get_columns('event_log') if column['name'] == 'user_id'
        )
        if not event_log_user_column.get('nullable', True) and db.engine.dialect.name == 'sqlite':
            db.session.execute(text('PRAGMA foreign_keys=OFF'))
            db.session.execute(text('ALTER TABLE event_log RENAME TO event_log_legacy'))
            db.session.execute(text('''
                CREATE TABLE event_log (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    camera_id INTEGER,
                    timestamp DATETIME,
                    source_name VARCHAR(100),
                    event_type VARCHAR(50),
                    description VARCHAR(500),
                    confidence FLOAT,
                    event_category VARCHAR(30),
                    detector VARCHAR(30),
                    severity VARCHAR(30),
                    event_metadata TEXT,
                    snapshot_path VARCHAR(255),
                    ip_address VARCHAR(45),
                    actor_name VARCHAR(150),
                    actor_email VARCHAR(150),
                    FOREIGN KEY(user_id) REFERENCES user (id),
                    FOREIGN KEY(camera_id) REFERENCES camera (id)
                )
            '''))
            db.session.execute(text('''
                INSERT INTO event_log
                    (id, user_id, camera_id, timestamp, source_name, event_type,
                     description, confidence, event_category, detector, severity,
                     event_metadata, snapshot_path, ip_address, actor_name, actor_email)
                SELECT id, user_id, camera_id, timestamp, source_name, event_type,
                       description, confidence, event_category, detector, severity,
                       event_metadata, snapshot_path, ip_address, actor_name, actor_email
                FROM event_log_legacy
            '''))
            db.session.execute(text('DROP TABLE event_log_legacy'))
            db.session.execute(text('PRAGMA foreign_keys=ON'))
            db.session.commit()


def create_admin_user(username, email, password):
    if not username or not email or not password:
        raise ValueError('Username, email, and password are required.')
    with app.app_context():
        ensure_database_schema()
        if User.query.filter((User.email == email) | (User.username == username)).first():
            raise ValueError('An account with that username or email already exists.')
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            role='admin',
            totp_secret=(encrypt_totp_secret(pyotp.random_base32())
                         if app.config.get('LEGACY_OTP') else None),
            totp_enabled=False,
        )
        db.session.add(new_user)
        db.session.commit()
        db.session.add(Settings(user_id=new_user.id, recipient_email=email))
        db.session.commit()
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'known_faces'), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'recordings'), exist_ok=True)
        return new_user
