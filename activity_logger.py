import json
import os
from datetime import datetime
import requests

class ActivityLogger:
    def __init__(self):
        self.log_file = "logs/activity.json"
        os.makedirs("logs", exist_ok=True)
        
    def get_public_ip(self):
        """Get user's public IP address"""
        try:
            response = requests.get('https://api.ipify.org?format=json', timeout=5)
            return response.json().get('ip', 'Unknown')
        except:
            return 'Unknown'
    
    def get_private_ip(self, request):
        """Get user's private IP address"""
        try:
            # Try different headers for private IP
            private_ip = (
                request.environ.get('HTTP_X_FORWARDED_FOR') or
                request.environ.get('HTTP_X_REAL_IP') or
                request.environ.get('REMOTE_ADDR') or
                'Unknown'
            )
            # If forwarded, get the first IP
            if ',' in private_ip:
                private_ip = private_ip.split(',')[0].strip()
            return private_ip
        except:
            return 'Unknown'
    
    def log_activity(self, username, activity, request, details=None):
        """Log user activity"""
        try:
            # Get IP addresses
            public_ip = self.get_public_ip()
            private_ip = self.get_private_ip(request)
            
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'username': username,
                'activity': activity,
                'public_ip': public_ip,
                'private_ip': private_ip,
                'user_agent': request.headers.get('User-Agent', 'Unknown'),
                'details': details or {}
            }
            
            # Load existing logs
            logs = []
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            # Add new log
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging activity: {e}")
    
    def get_logs(self, limit=100, username=None):
        """Get activity logs"""
        try:
            if not os.path.exists(self.log_file):
                return []
            
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Filter by username if specified
            if username:
                logs = [log for log in logs if log.get('username') == username]
            
            # Return latest logs
            return logs[-limit:][::-1]  # Reverse to show newest first
            
        except Exception as e:
            print(f"Error getting logs: {e}")
            return []
    
    def get_stats(self):
        """Get activity statistics"""
        try:
            logs = self.get_logs(limit=1000)
            
            if not logs:
                return {}
            
            # Calculate stats
            total_activities = len(logs)
            unique_users = len(set(log.get('username') for log in logs))
            unique_ips = len(set(log.get('public_ip') for log in logs))
            
            # Activity counts
            activity_counts = {}
            for log in logs:
                activity = log.get('activity', 'Unknown')
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
            # Recent activities (last 24 hours)
            from datetime import datetime, timedelta
            yesterday = datetime.now() - timedelta(days=1)
            recent_logs = [
                log for log in logs 
                if datetime.fromisoformat(log.get('timestamp', '')) > yesterday
            ]
            
            return {
                'total_activities': total_activities,
                'unique_users': unique_users,
                'unique_ips': unique_ips,
                'activity_counts': activity_counts,
                'recent_activities_24h': len(recent_logs),
                'last_activity': logs[0] if logs else None
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}