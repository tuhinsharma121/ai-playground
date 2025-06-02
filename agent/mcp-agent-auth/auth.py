from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2AuthorizationCodeBearer
from jwt import PyJWKClient

from utils.constants import constants
from utils.pylogger import get_python_logger

logger = get_python_logger()

# OAuth2 scheme for authorization
oauth_2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{constants.JWT_SSO_BASE_URL}/protocol/openid-connect/auth",
    tokenUrl=f"{constants.JWT_SSO_BASE_URL}/protocol/openid-connect/token",
)

# Initialize the JWK client
url = f"{constants.JWT_SSO_BASE_URL}/protocol/openid-connect/certs"
optional_custom_headers = {"User-agent": "custom-user-agent"}
jwks_client = PyJWKClient(url, headers=optional_custom_headers)

# FatsAPI Setup
tags_metadata = [
    {"name": "RHSC AI Assistant", "description": "Ask RHSC AI Assistant."},
    {"name": "Heartbeat", "description": "Check if server is running fine."},
]

app = FastAPI(
    title="RHSC AI Assistant",
    description="APIs for RHSC AI Assistant",
    docs_url="/docs",
    redoc_url=None,
    version="1.0.0",
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/schemas/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# This route serves as the heartbeat or health check endpoint for the application.
@app.get("/", response_model=dict, tags=["Heartbeat"])
def home_page():
    response = dict()
    response["message"] = app.description
    response["status"] = status.HTTP_200_OK
    return response


def valid_access_token(access_token: Annotated[str, Depends(oauth_2_scheme)]):
    """
    Validates the provided access token using JWKS.

    Parameters
    ----------
    access_token : str
        The access token to validate.

    Returns
    -------
    dict
        The decoded token data if validation is successful.

    Raises
    ------
    HTTPException
        If the token is invalid, expired, or if there is an error during validation.
    """
    try:
        logger.info(f"Authentication URL : {url}")
        # Retrieve the signing key from the JWKS client
        signing_key = jwks_client.get_signing_key_from_jwt(access_token)

        # Decode the access token using the signing key
        data = jwt.decode(
            access_token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_exp": constants.JWT_EXPIRY, "verify_aud": False},
        )
        return data
    except jwt.exceptions.DecodeError:
        # Raised if the token cannot be decoded
        raise HTTPException(status_code=401, detail="Error decoding token")
    except jwt.exceptions.ExpiredSignatureError:
        # Raised if the token has expired
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.exceptions.InvalidTokenError:
        # Raised if the token is invalid
        raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.exceptions.PyJWKClientError as e:
        # Raised if there is an error with the JWK client
        raise HTTPException(status_code=401, detail="JWK Client Error: " + str(e))
    except Exception as e:
        # Raised if there is a general error during validation
        raise HTTPException(
            status_code=401, detail="Could not validate credentials: " + str(e)
        )


@app.post(
    "/api/v1/rhsc-ai-assistant-conversational-answer",
    tags=["RHSC AI Assistant"],
    summary="Conversational - Response Generation",
    dependencies=[Depends(valid_access_token)],
)
def conversational_answer(n: str):
    return {"ok": n}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
