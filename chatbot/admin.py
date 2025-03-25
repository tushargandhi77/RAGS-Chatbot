from django.contrib import admin
from .models import Document
from .rag_utils import process_document, clear_collection

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('uploaded_at',)

    def save_model(self, request, obj, form, change):
        # Clear previous data in AstraDB before uploading new document
        clear_collection()
        
        # Save the new document to the Django filesystem
        super().save_model(request, obj, form, change)
        
        # Process and upload to AstraDB
        file_path = obj.file.path
        process_document(file_path)

    def has_add_permission(self, request):
        # Allow only one document at a time
        return Document.objects.count() < 1 or Document.objects.count() == 0

    def has_delete_permission(self, request, obj=None):
        # Allow deletion to upload a new document
        return True