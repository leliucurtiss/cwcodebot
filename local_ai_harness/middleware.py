class LoggingMiddleware:
    def __init__(self, app, logger):
        self.app = app
        self.logger = logger

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        
        status_code = [None]
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.error(f"Request failed: {method} {path} - Error: {e}")
            raise e
        finally:
            if status_code[0] and status_code[0] >= 400:
                self.logger.error(f"Error Response: {method} {path} - Status: {status_code[0]}")
