import numpy as np
import cv2  # type: ignore

from genetic_algorithm.GeneticAlgorithm import GeneticAlgorithm

from sklearn.metrics import jaccard_score  # type: ignore


def mask_tgi(img: np.ndarray, b: float, r: float, threshold: float) -> np.ndarray:
    B: np.ndarray = img[:, :, 0]
    G: np.ndarray = img[:, :, 1]
    R: np.ndarray = img[:, :, 2]

    TGI: np.ndarray = b * B + G + r * R

    mask: np.ndarray = np.ones((img.shape[0], img.shape[1]))
    mask *= TGI

    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    return mask


def get_tgi(img: np.ndarray, mask: np.ndarray,
            mut_force: float = 0.5, prob_mut: float = 0.75, internal_mut_bound: float = 1e-7,
            max_iter: int = 75, delta_converged: float = 1e-4, population_count: int = 20,
            initial_population: np.ndarray | None = None,
            initial_population_bound: float | np.ndarray = np.array([1, 1, 7]),
            iter_increase: int = 30, increase_force: float = 0.2, decrease_increase_force: float = 0.5,
            silent: bool = False) -> GeneticAlgorithm:
    dim: int = 3
    gen_alg: GeneticAlgorithm = GeneticAlgorithm(lambda b, r, thr:
                                                 jaccard_score(mask.reshape(mask.size).clip(0, 1),
                                                               mask_tgi(img, b, r, thr).reshape(mask.size)),
                                                 mut_force, prob_mut, internal_mut_bound, max_iter, delta_converged,
                                                 population_count, dim, initial_population, initial_population_bound,
                                                 iter_increase, increase_force, decrease_increase_force, silent)
    gen_alg.run()
    return gen_alg
