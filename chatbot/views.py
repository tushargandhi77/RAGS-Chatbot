from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .rag_utils import get_response

@csrf_exempt
def chat(request):
    if request.method == "POST":
        query = request.POST.get("query")
        if query:
            response = get_response(query)
            return JsonResponse({"response": response})
        return JsonResponse({"error": "Query is required"}, status=400)
    return JsonResponse({"error": "Invalid request"}, status=400)