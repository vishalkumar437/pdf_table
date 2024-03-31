from django.http import HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import segmentace_textu_V1 as st
import os
import shutil

@csrf_exempt
def hello(request):
    print("hi")
    return HttpResponse("Hello, World!")

@csrf_exempt
def index(request):
    try:
        # Basic Setup
        if request.method == 'POST':
            print(request)
            pdf_file = request.FILES.get('pdf', None)
            print(pdf_file)
            if pdf_file is None:
                return HttpResponseBadRequest('No PDF file in request')
            # if pdf file is there copy it to the server
            with open('pdf_file.pdf', 'wb+') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)
            excepzip = st.main('pdf_file.pdf')
            if excepzip is None:
                return HttpResponseBadRequest('No tables found in the PDF')
            # return binary file excel.zip
            if excepzip:
                if os.path.exists('excel'):
                    shutil.make_archive('excel', 'zip', 'excel')
            with open('excel.zip', 'rb') as file:
                return HttpResponse(file.read(), content_type='application/zip')
        else:
            return HttpResponseBadRequest('Only POST method is allowed')

    except Exception as e:
        msg = f'An exception occurred - {e} ({type(e)})'
        return HttpResponse(msg)
