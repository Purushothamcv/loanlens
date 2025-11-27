import subprocess
import sys
import socket

def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False

def main():
    host = "127.0.0.1"
    preferred_ports = [8000, 8001, 8002, 8003, 8004, 8005]
    
    print("🔍 Checking for available ports...")
    
    for port in preferred_ports:
        if is_port_available(host, port):
            print(f"✅ Port {port} is available!")
            print(f"🚀 Starting server on {host}:{port}")
            
            # Set environment variable for the port
            import os
            os.environ['PORT'] = str(port)
            
            # Start the server
            try:
                subprocess.run([sys.executable, 'api.py'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error starting server: {e}")
            except KeyboardInterrupt:
                print("\n👋 Server stopped by user")
            return
    
    print("❌ No available ports found in range 8000-8005")
    print("💡 Please stop other services or restart your computer")

if __name__ == "__main__":
    main()
