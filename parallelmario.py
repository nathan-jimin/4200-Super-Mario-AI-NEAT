import retro
import numpy as np
import cv2
import neat
import pickle

class Worker(object):
	
	def __init__(self, genome, config):
		self.genome = genome
		self.config = config

	def work(self):

		self.env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2')

		self.env.reset()


		ob, _, _, _ = self.env.step(self.env.action_space.sample())

		inx = int(ob.shape[0]/8)
		iny = int(ob.shape[1]/8)

		done = False

		net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

		imgarray = []

		fitness = 0

		xpos = 0
		xpos_max = 0

		endoflevel = 0

		counter = 0

		while not done:

			#to see what is going on (BUG: wont close windows on done)
			#self.env.render()

			#convert image to matrix, need to reshape and resize to save computing power
			ob = cv2.resize(ob, (inx, iny))

			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))

			imgarray = np.ndarray.flatten(ob)

			actions = net.activate(imgarray)

			ob, rew, done, info = self.env.step(actions)

			xpos = info['xpos']
			endoflevel = info['endoflevel']

			if xpos > xpos_max:
				xpos_max = xpos
				counter = 0
				fitness += 1
			else:
				counter += 1

			if counter > 250:
				done = True

			if endoflevel > 0:
				fitness += 10000
				done = True

		self.viewer.close()
		self.env.close()
		print("fitness =", fitness)
		return fitness

def eval_genomes(genome, config):
	worky = Worker(genome, config)
	return worky.work()



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
	neat.DefaultSpeciesSet, neat.DefaultStagnation, 
	'config-feedforward-mario')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pk1', 'wb') as output:
	pickle.dump(winner, output, 1)
