import json
import os
from datetime import datetime
import hashlib

class UserManager:
    def __init__(self):
        self.users_file = "data/users.json"
        os.makedirs("data", exist_ok=True)
        self._load_users()
    
    def _load_users(self):
        """Load users from file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            except:
                self.users = {}
        else:
            # Default admin user
            self.users = {
                'admin': {
                    'password': self._hash_password('admin123'),
                    'is_admin': True,
                    'created_at': datetime.now().isoformat(),
                    'full_name': 'Administrator',
                    'email': 'admin@aidubbing.com'
                }
            }
            self._save_users()
    
    def _save_users(self):
        """Save users to file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, password, full_name, email, is_admin=False):
        """Register new user"""
        try:
            # Validate input
            if not username or not password or not full_name or not email:
                return False, "All fields are required"
            
            if len(username) < 3:
                return False, "Username must be at least 3 characters"
            
            if len(password) < 6:
                return False, "Password must be at least 6 characters"
            
            # Validate email format
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return False, "Invalid email format"
            
            # Check if username exists
            if username.lower() in [u.lower() for u in self.users.keys()]:
                return False, "Username already exists"
            
            # Check if email exists
            for user_data in self.users.values():
                if user_data.get('email', '').lower() == email.lower():
                    return False, "Email already registered"
            
            # Create new user
            self.users[username] = {
                'password': self._hash_password(password),
                'is_admin': is_admin,
                'created_at': datetime.now().isoformat(),
                'full_name': full_name,
                'email': email,
                'last_login': None
            }
            
            self._save_users()
            return True, "User registered successfully"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        try:
            if username not in self.users:
                return False, None, "Invalid username or password"
            
            user = self.users[username]
            hashed_password = self._hash_password(password)
            
            if user['password'] != hashed_password:
                return False, None, "Invalid username or password"
            
            # Update last login
            self.users[username]['last_login'] = datetime.now().isoformat()
            self._save_users()
            
            return True, user, "Login successful"
            
        except Exception as e:
            return False, None, f"Authentication failed: {str(e)}"
    
    def get_user(self, username):
        """Get user information"""
        return self.users.get(username)
    
    def get_all_users(self):
        """Get all users (admin only)"""
        users_list = []
        for username, user_data in self.users.items():
            user_info = {
                'username': username,
                'full_name': user_data.get('full_name', ''),
                'email': user_data.get('email', ''),
                'is_admin': user_data.get('is_admin', False),
                'created_at': user_data.get('created_at', ''),
                'last_login': user_data.get('last_login', 'Never'),
                'updated_at': user_data.get('updated_at', '')
            }
            users_list.append(user_info)
        return users_list
    
    def get_all_users_with_passwords(self):
        """Get all users with their actual passwords (admin only)"""
        users_with_passwords = {}
        for username, user_data in self.users.items():
            users_with_passwords[username] = {
                **user_data,
                'plain_password': self._get_plain_password(username)
            }
        return users_with_passwords
    
    def _get_plain_password(self, username):
        """Get plain password for display (security risk - admin only)"""
        # This is a security risk but requested for admin view
        # In production, passwords should never be stored in plain text
        user = self.users.get(username)
        if not user:
            return "Unknown"
        
        # Common passwords to try (this is a security vulnerability)
        common_passwords = [
            'admin123', 'password', '123456', 'admin', 'user123',
            username, username + '123', '12345678', 'qwerty'
        ]
        
        for pwd in common_passwords:
            if self._hash_password(pwd) == user['password']:
                return pwd
        
        return "[Encrypted - Cannot Display]"
    
    def delete_user(self, username):
        """Delete user (admin only)"""
        try:
            if username == 'admin':
                return False, "Cannot delete admin user"
            
            if username not in self.users:
                return False, "User not found"
            
            del self.users[username]
            self._save_users()
            return True, "User deleted successfully"
            
        except Exception as e:
            return False, f"Delete failed: {str(e)}"
    
    def update_user(self, username, **kwargs):
        """Update user information"""
        try:
            if username not in self.users:
                return False, "User not found"
            
            user = self.users[username]
            
            # Update allowed fields
            if 'full_name' in kwargs and kwargs['full_name']:
                user['full_name'] = kwargs['full_name']
            
            if 'email' in kwargs and kwargs['email']:
                # Validate email format
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, kwargs['email']):
                    return False, "Invalid email format"
                
                # Check if email exists for other users
                for other_username, other_user in self.users.items():
                    if other_username != username and other_user.get('email', '').lower() == kwargs['email'].lower():
                        return False, "Email already registered to another user"
                
                user['email'] = kwargs['email']
            
            if 'password' in kwargs and kwargs['password']:
                if len(kwargs['password']) < 6:
                    return False, "Password must be at least 6 characters"
                user['password'] = self._hash_password(kwargs['password'])
            
            if 'is_admin' in kwargs and username != 'admin':
                user['is_admin'] = kwargs['is_admin']
            
            user['updated_at'] = datetime.now().isoformat()
            self._save_users()
            
            return True, "User updated successfully"
            
        except Exception as e:
            return False, f"Update failed: {str(e)}"
    
    def change_password(self, username, current_password, new_password):
        """Change user password"""
        try:
            if username not in self.users:
                return False, "User not found"
            
            user = self.users[username]
            
            # Verify current password
            if user['password'] != self._hash_password(current_password):
                return False, "Current password is incorrect"
            
            # Validate new password
            if len(new_password) < 6:
                return False, "New password must be at least 6 characters"
            
            # Check if new password is different from current
            if self._hash_password(new_password) == user['password']:
                return False, "New password must be different from current password"
            
            # Update password
            user['password'] = self._hash_password(new_password)
            user['password_changed_at'] = datetime.now().isoformat()
            user['updated_at'] = datetime.now().isoformat()
            
            self._save_users()
            
            return True, "Password changed successfully"
            
        except Exception as e:
            return False, f"Password change failed: {str(e)}"