from django.contrib import admin
from .models import Sentiment
# Register your models here.

@admin.register(Sentiment)
class ProfanityAdmin(admin.ModelAdmin):
    list_display = ['id', 'mytext', 'sentiment']