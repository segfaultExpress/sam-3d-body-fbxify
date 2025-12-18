"""
Internationalization (i18n) module for FBXify app.
Provides translation loading and language detection.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Default language
DEFAULT_LANGUAGE = "en"

# Supported languages
SUPPORTED_LANGUAGES = ["en", "ja"]

# Translation cache
_translations: Dict[str, Dict[str, Any]] = {}


def get_i18n_dir() -> Path:
    """Get the i18n directory path."""
    return Path(__file__).parent


def load_translations(lang: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """
    Load translations for a given language.
    
    Args:
        lang: Language code (e.g., 'en', 'ja')
        
    Returns:
        Dictionary containing translations for the specified language
    """
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    
    # Return cached translations if available
    if lang in _translations:
        return _translations[lang]
    
    # Load translations from JSON file
    i18n_dir = get_i18n_dir()
    lang_file = i18n_dir / f"{lang}.json"
    
    if not lang_file.exists():
        # Fallback to default language if file doesn't exist
        if lang != DEFAULT_LANGUAGE:
            return load_translations(DEFAULT_LANGUAGE)
        return {}
    
    try:
        with open(lang_file, "r", encoding="utf-8") as f:
            translations = json.load(f)
            _translations[lang] = translations
            return translations
    except Exception as e:
        print(f"Warning: Failed to load translations for {lang}: {e}")
        if lang != DEFAULT_LANGUAGE:
            return load_translations(DEFAULT_LANGUAGE)
        return {}


def get_translation(key: str, lang: str = DEFAULT_LANGUAGE, default: Optional[str] = None) -> str:
    """
    Get a translation for a given key.
    
    Args:
        key: Translation key (supports dot notation, e.g., "ui.profile")
        lang: Language code
        default: Default value if key is not found
        
    Returns:
        Translated string or default value
    """
    translations = load_translations(lang)
    
    # Navigate through nested dictionary using dot notation
    keys = key.split(".")
    value = translations
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default if default is not None else key
    
    return value if isinstance(value, str) else (default if default is not None else key)


def format_translation(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs) -> str:
    """
    Get a translation and format it with provided arguments.
    
    Args:
        key: Translation key
        lang: Language code
        **kwargs: Format arguments
        
    Returns:
        Formatted translated string
    """
    translation = get_translation(key, lang)
    try:
        return translation.format(**kwargs)
    except (KeyError, ValueError):
        # If formatting fails, return the translation as-is
        return translation


class Translator:
    """Translation helper class for easier usage."""
    
    def __init__(self, lang: str = DEFAULT_LANGUAGE):
        self.lang = lang
        self._translations = load_translations(lang)
    
    def t(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        Translate a key.
        
        Args:
            key: Translation key (supports dot notation)
            default: Default value if key is not found
            **kwargs: Format arguments
            
        Returns:
            Translated and formatted string
        """
        translation = get_translation(key, self.lang, default)
        if kwargs:
            try:
                return translation.format(**kwargs)
            except (KeyError, ValueError):
                return translation
        return translation
    
    def get(self, key: str, default: Optional[str] = None) -> Any:
        """
        Get a translation value (can be any type, not just string).
        
        Args:
            key: Translation key (supports dot notation)
            default: Default value if key is not found
            
        Returns:
            Translation value
        """
        translations = self._translations
        
        keys = key.split(".")
        value = translations
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default if default is not None else key
        
        return value
