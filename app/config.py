from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    # Delai maximal d'une requete parcelles a la base. 60 s par defaut : la
    # requete complete (37 colonnes, JSONB volumineux) charge toutes les
    # parcelles et peut depasser 20 s sur une base distante (Neon) ou un lien
    # lent, surtout au reveil de la base. Au-dela du delai, le cache renvoie une
    # erreur 503 au lieu de servir les donnees.
    parcels_query_timeout_s: float = 60.0
    gee_service_account_key: str = "{}"
    hf_model_url: str = ""
    # Poids YOLO de segmentation de parcelles embarqué (vide : app/data/models/field_segmentation).
    # Le modèle est exécuté dans le processus backend, sans API ni service séparé.
    field_segmentation_model_path: str = ""
    # Seuil du masque quand la requête n'en fournit pas. 0,1 = confiance de production
    # du projet source : au-delà, le masque devient quasi vide (confiances médianes ~0,2).
    field_segmentation_threshold: float = 0.1
    # Préchauffage des modèles au démarrage (import torch + premier passage).
    # Passer à false en développement pour ne pas payer ~20 s à chaque --reload.
    warmup_models: bool = True
    vhr_model_url: str = "http://localhost:8010"
    # Auto-start du service de tracage VHR (serve_vhr.py, modele delineate_jecam_v3_precision.pt).
    # vhr_service_dir vide : pas d'auto-start (le service doit etre lance a la main).
    # vhr_service_python vide : utilise le venv du service s'il existe, sinon le python courant.
    vhr_service_dir: str = ""
    vhr_service_python: str = ""
    vhr_service_port: int = 8010
    # Cache disque des zones tracees par le service VHR (vide : AGRISAT-BACKEND/.cache/vhr_parcels).
    vhr_cache_dir: str = ""
    vhr_cache_ttl_days: float = 30.0
    google_maps_api_key: str = ""
    # Detection de parcelles par Gemini (POST /api/gemini-parcels). La cle Gemini
    # est distincte de la cle Google Maps : une cle Maps n'a pas acces a l'API
    # Generative Language (403 API_KEY_SERVICE_BLOCKED). A defaut de cle Gemini,
    # google_maps_api_key est tentee en repli.
    gemini_api_key: str = ""
    # Modele Gemini utilise par POST /api/gemini-parcels. `gemini-2.5-flash`
    # n'est plus propose aux nouveaux comptes : 3.8-flash est le modele courant.
    gemini_model: str = "gemini-3.8-flash"
    # Mode de detection : "image" (defaut — Gemini DESSINE les limites en rouge
    # sur la photo, que le backend reconvertit en polygones ; les contours epousent
    # la structure reelle) ou "text" (Gemini enumere des coordonnees JSON, ce qui
    # reste approximatif : ancien comportement, conserve en repli).
    gemini_parcels_mode: str = "image"
    # Modele image utilise quand gemini_parcels_mode = "image".
    gemini_image_model: str = "gemini-nano-banana-2.1"
    # Largeur (m) de la zone capturee autour du centre de la carte. Le zoom
    # Google Static Maps est plafonne a 21 : pour garder le detail metrique d'une
    # tuile de 20 m sur une zone de 100 m, la capture est une mosaique 5x5.
    gemini_capture_span_m: float = 400.0
    gemini_capture_tile_span_m: float = 120.0
    gemini_capture_max_tiles: int = 5
    # Taille logique d'une tuile Static Maps (max 640) ; scale=2 double les pixels
    # pour une capture simple, scale=1 suffit pour les tuiles de mosaique.
    gemini_capture_size_px: int = 640
    gemini_timeout_s: float = 150.0
    airbus_oneatlas_api_key: str = ""
    port: int = 3001
    host: str = "0.0.0.0"
    frontend_origin: str = ""
    planet_api_key: str = ""


settings = Settings()
