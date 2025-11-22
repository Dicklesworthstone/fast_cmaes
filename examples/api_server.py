"""Minimal REST API exposing fastcma via FastAPI/uvicorn.

Run:
    uvicorn examples.api_server:app --reload

POST /optimize with JSON:
{
  "x0": [0.5, 0.5],
  "sigma": 0.4,
  "maxfevals": 5000
}
Returns best x and f for a sphere objective.
"""
from typing import List
from fastapi import FastAPI
import fastcma

app = FastAPI(title="fastcma API", version="0.1.4")


def sphere(x: List[float]) -> float:
    return sum(v * v for v in x)


@app.post("/optimize")
async def optimize(x0: List[float], sigma: float = 0.4, maxfevals: int = 5000):
    xmin, es = fastcma.fmin(sphere, x0, sigma=sigma, maxfevals=maxfevals)
    xbest, fbest, _evals_best, counteval, _iters, _xmean, _stds = es.result
    return {
        "fbest": fbest,
        "xmin": xmin,
        "evals": counteval,
        "dims": len(xmin),
    }
