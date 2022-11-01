from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from sys import stdout
from urllib.parse import parse_qs, urlparse
import cv2
import numpy as np
from sift import SIFT
from os import getenv


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            path = urlparse(self.path)
            query = parse_qs(path.query)

            gray = query['gray'] if 'gray' in query else None
            descriptor = query['descriptor'] if 'descriptor' in query else None

            content_length = int(self.headers['content-length'] or 0)

            if (content_length == 0):
                self.send_response(500)
                return

            body = self.rfile.read(content_length)

            sift = SIFT()

            img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), 1)
            if (gray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = SIFT()
            keypoints = sift.detect(img)

            descriptors = None
            if (descriptor == 'numbers'):
                _, descriptors = sift.compute(img, keypoints)

            def p(i: int):
                keypoint = keypoints[i]
                _descriptor = None if descriptors is None else descriptors[i]

                d = {
                    'pt': keypoint.pt,
                    'size': keypoint.size,
                    'angle': keypoint.angle,
                    'response': keypoint.response,
                    'octave': keypoint.octave,
                }

                if (_descriptor is not None):
                    d['descriptor'] = _descriptor.tolist()

                return d

            json_s = json.dumps({
                'keypoints_count': len(keypoints),
                'keypoints': list(map(p, range(len(keypoints)))),
            }).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json_s)

        except Exception as e:
            self.send_response(500)
            raise e


def main():
    port = int(getenv('PORT') or 0)
    server = HTTPServer(('127.0.0.1', port), Handler)
    json.dump({'port': server.server_address[1]}, stdout)
    print()
    server.serve_forever()


if __name__ == '__main__':
    main()
