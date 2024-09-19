from django.http import HttpResponse
from psynamic.models import Study
from django.template import loader
from django.shortcuts import render

# Create your views here.
def index(request):
    studies = Study.objects.all()
    context = {
        'studies': studies
    }
    template = loader.get_template('psynamic/index.html')
    return render(request, "psynamic/index.html", context)