from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')

def getPredictions(URL_length, Geolocation, top_level_domain, Javascript_length,Javascript_obfuscate_length, Who_is, HTTPS):
    
    #loading all of the stored objects
    model = pickle.load(open('model.sav', 'rb'))
    scaled = pickle.load(open('scaling.sav', 'rb'))
    dic_geolocation = pickle.load(open('geolocation_dictionary.sav', 'rb'))
    dic_tld = pickle.load(open('tld_dictionary.sav', 'rb'))
    
    #When the user will provide country so it will be categorical,so with the help of this dictionary it will encode it into numeric form
    converted_geo=dic_geolocation[Geolocation]
    
    #When the user will provide tld so it will be categorical,so with the help of this dictionary it will encode it into numeric form
    converted_tld=dic_tld[top_level_domain]
    
    #When the user will provide Who_is so it will be categorical,so it will encode it into numeric form
    if (Who_is =='complete' or Who_is =='Complete' or Who_is =='COMPLETE') :
        Who_is=0
    elif (Who_is =='incomplete' or Who_is =='Incomplete' or Who_is =='INCOMPLETE'):
        Who_is=1      
    
    #When the user will provide HTTPS so it will be categorical,so it will encode it into numeric form    
    if (HTTPS =='yes' or HTTPS =='HTTPS' or HTTPS =='Yes' or HTTPS =='Yes') :
        HTTPS=1
    elif (HTTPS=='no' or HTTPS =='No' or HTTPS =='NO' or HTTPS =='HTTP'):
        HTTPS=0    
        
    #Prediction after scaling of numeric columns  
    prediction = model.predict(scaled.transform([
        [URL_length, converted_geo,  converted_tld, Who_is, HTTPS, Javascript_length,Javascript_obfuscate_length]]))
    
    
    if prediction == 0:
        return 'malicious'
    elif prediction == 1:
        return 'benign'
    else:
        return 'error'

#Results for WebPage
def result(request):
    URL_length = float(request.GET['URL_length'])
    Geolocation = request.GET['Geolocation']
    top_level_domain = request.GET['top_level_domain']
    Javascript_length = float(request.GET['Javascript_length'])
    Javascript_obfuscate_length = float(request.GET['Javascript_obfuscate_length'])
    Who_is = request.GET['Who_is']
    HTTPS = request.GET['HTTPS']

    result = getPredictions(URL_length, Geolocation, top_level_domain, Javascript_length,Javascript_obfuscate_length, Who_is, HTTPS)

    return render(request, 'result.html', {'result': result})
