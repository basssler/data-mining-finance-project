from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class FileSaveHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        filename = self.path[1:] # e.g., /WMT_2023.csv
        if not filename:
            filename = 'dump.txt'
        
        filepath = os.path.join(r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\raw\capital_iq\key_developments', filename)
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        with open(filepath, 'wb') as f:
            f.write(post_data)
            
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b"Success")
        print(f"Saved {filepath}")

def run(server_class=HTTPServer, handler_class=FileSaveHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
