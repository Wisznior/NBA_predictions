from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django import forms

class RegistrationForm(UserCreationForm):
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={"placeholder": "Enter your password"}))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={"placeholder": "Confirm your password"}))
    class Meta:
        model = User
        fields = ("first_name", "last_name", "username", "email", "password1", "password2")
        widgets = {
            "first_name": forms.TextInput(attrs={"placeholder": "Enter your first name"}),
            "last_name": forms.TextInput(attrs={"placeholder": "Enter your last name"}),
            "username": forms.TextInput(attrs={"placeholder": "Enter your username"}),
            "email": forms.EmailInput(attrs={"placeholder": "Enter your email"}),
        }

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={"placeholder": "Enter your username"}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={"placeholder": "Enter your password"}))
    class Meta:
        fields = ("username", "password")
