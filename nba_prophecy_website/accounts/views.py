from django.shortcuts import render, redirect
from .forms import RegistrationForm


def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegistrationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    return render(request, "login.html", {})
