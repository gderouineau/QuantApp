from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse

ALLOWED_PREFIXES = ("/accounts/", "/admin/", "/static/", "/favicon.ico")

class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path
        if request.user.is_authenticated or any(path.startswith(p) for p in ALLOWED_PREFIXES):
            return self.get_response(request)
        return redirect(f"{settings.LOGIN_URL}?next={path}")
