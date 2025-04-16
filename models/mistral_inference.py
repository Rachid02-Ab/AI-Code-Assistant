import requests
import json

def get_mistral_response(prompt):
    # Configuration (à externaliser en production)
    api_key = "onYcoSEMsQRVRfrt7WfjQrwrMcKJ8HzE"  # Clé exemple - remplacez par votre vraie clé
    endpoint = "https://api.mistral.ai/v1/chat/completions"  # Endpoint moderne
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],  # Format messages requis
        "temperature": 0.7,
        "max_tokens": 800,
        "top_p": 1.0,
        "stream": False
    }
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,  # Utilisation de json= pour la sérialisation automatique
        )
        
        # Vérification du statut HTTP
        response.raise_for_status()
        response_data = response.json()
        
        # Extraction du contenu de la réponse
        if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur API: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Détails de l'erreur: {e.response.text}")
        return None

 