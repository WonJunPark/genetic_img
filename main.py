import cv2, random, os, sys
import numpy as np
from copy import deepcopy
from skimage.measure import compare_mse
import multiprocessing as mp

img = cv2.imread('img/luffy.jpg')
height, width, channels = img.shape

# hyperparameters
# 첫번째 유전자 개수
n_initial_genes = 50
# 한 세대당 유전자 그룹의 숫자
n_population = 50
# 돌연변이 발생 확률
prob_mutation = 0.01
# 유전자 그룹의 원이 추가될 확률 0.3
prob_add = 0.3
# 유전자 그룹의 원이 사라질 확률 0.2
prob_remove = 0.2

# 원의 크기
min_radius, max_radius = 3, 10
# 이미지 저장 주기
save_every_n_iter = 100

# Gene
class Gene():
  def __init__(self):
    # 센터, 반지름, 색깔 초기화
    self.center = np.array([random.randint(0, width), random.randint(0, height)])
    self.radius = random.randint(min_radius, max_radius)
    self.color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

  def mutate(self):
    # 변이의 크기, 평균 15 표준편차 4
    # 15% 정도 변이를 줌
    mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100

    r = random.uniform(0, 1)
    if r < 0.33: # radius
      self.radius = np.clip(random.randint(
        int(self.radius * (1 - mutation_size)),
        int(self.radius * (1 + mutation_size))
      ), 1, 100)
    elif r < 0.66: # center
      self.center = np.array([
        np.clip(random.randint(
          int(self.center[0] * (1 - mutation_size)),
          int(self.center[0] * (1 + mutation_size))),
        0, width),
        np.clip(random.randint(
          int(self.center[1] * (1 - mutation_size)),
          int(self.center[1] * (1 + mutation_size))),
        0, height)
      ])
    else: # color
      self.color = np.array([
        np.clip(random.randint(
          int(self.color[0] * (1 - mutation_size)),
          int(self.color[0] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[1] * (1 - mutation_size)),
          int(self.color[1] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[2] * (1 - mutation_size)),
          int(self.color[2] * (1 + mutation_size))),
        0, 255)
      ])

# compute fitness
# 얼마만큼 원본 이미지의 가까운지 확인
def compute_fitness(genome):
  out = np.ones((height, width, channels), dtype=np.uint8) * 255

  # 유전자의 값을 원으로 그림
  for gene in genome:
    cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(int(gene.color[0]), int(gene.color[1]), int(gene.color[2])), thickness=-1)

  # mean squared error
  # 두 이미지의 차이
  fitness = 255. / compare_mse(img, out)

  return fitness, out

# compute population
# 유전자를 한꺼번에 돌연변이로 만듬
def compute_population(g):
  genome = deepcopy(g)
  # mutation
  # 유전자의 개수에 따라 변이를 바꿔줌
  if len(genome) < 200:
    for gene in genome:
      if random.uniform(0, 1) < prob_mutation:
        gene.mutate()
  else:
    for gene in random.sample(genome, k=int(len(genome) * prob_mutation)):
      gene.mutate()

  # add gene
  # 유전자 추가
  if random.uniform(0, 1) < prob_add:
    genome.append(Gene())

  # remove gene
  # 유전자 삭제
  if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
    genome.remove(random.choice(genome))

  # compute fitness
  # 새로운 유전자의 점수 측정
  new_fitness, new_out = compute_fitness(genome)

  return new_fitness, genome, new_out

# main
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)

  # 병렬 처리
  p = mp.Pool(mp.cpu_count() - 1)

  # 1st gene
  best_genome = [Gene() for _ in range(n_initial_genes)]

  best_fitness, best_out = compute_fitness(best_genome)

  n_gen = 0

  while True:
    try:
      results = p.map(compute_population, [deepcopy(best_genome)] * n_population)
    except KeyboardInterrupt:
      p.close()
      break

    results.append([best_fitness, best_genome, best_out])

    new_fitnesses, new_genomes, new_outs = zip(*results)

    best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)

    best_fitness, best_genome, best_out = best_result[0]

    # end of generation
    print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
    n_gen += 1

    # visualize
    if n_gen % save_every_n_iter == 0:
      cv2.imwrite('result/%s_%s.jpg' % ('result2', n_gen), best_out)

    cv2.imshow('best out', best_out)
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break

  cv2.imshow('best out', best_out)
  cv2.waitKey(0)