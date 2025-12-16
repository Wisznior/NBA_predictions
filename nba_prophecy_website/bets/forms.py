from django import forms

class BetForm(forms.Form):
    game_id = forms.IntegerField(widget=forms.HiddenInput)

    home_score = forms.IntegerField(min_value=0, max_value=999)
    visitor_score = forms.IntegerField(min_value=0, max_value=999)