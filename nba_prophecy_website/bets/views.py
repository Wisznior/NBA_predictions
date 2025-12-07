from django.shortcuts import render

def bets_view(request):
    return render(request, "bets.html", {})
