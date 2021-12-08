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

		self.env = retro.make('SuperMarioWorld-Snes', 'DonutPlains1')

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
		xpos_prev = 0

		ypos = 0 
		ypos_prev = 0 
		ypos_max = 0 
		endoflevel = 0

		counter = 0

		dead = 0

		while not done:
			#to see what is going on	
			img = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)	
			cv2.imshow("training", img)
			cv2.waitKey(2)
			
			#convert image to matrix, need to reshape and resize to save computing power
			ob = cv2.resize(ob, (inx, iny))

			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))

			imgarray = np.ndarray.flatten(ob)
			#imgarray = np.interp(imgarray, (0, 254), (-1, 1))

			actions = net.activate(imgarray)

			ob, rew, done, info = self.env.step(actions)

			xpos = info['xpos']
			ypos = info['ypos']
			endoflevel = info['endoflevel']
			dead = info['dead']

			if xpos > xpos_max:
				xpos_max = xpos
				fitness += 1

			if xpos > xpos_prev:
				xpos_prev = xpos
				fitness += 1
				counter = 0

			if ypos > ypos_max:
				ypos_max = ypos
				fitness += 5

			if ypos > ypos_prev:
				fitness += 1
				ypos_prev = ypos

			else:
				counter += 1

			if counter > 120:
				done = True

			if dead == 9:
				done = True

			if endoflevel > 0:
				fitness += 100000
				done = True

		#self.viewer.close()
		#self.env.close()
		print("fitness =", fitness)
		cv2.destroyAllWindows()
		return fitness

def eval_genomes(genome, config):
	worky = Worker(genome, config)
	return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
	neat.DefaultSpeciesSet, neat.DefaultStagnation, 
	'config-feedforward-mario')

p = neat.Population(config)

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-217')
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-12')

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))



pe = neat.ParallelEvaluator(5, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pk1', 'wb') as output:
	pickle.dump(winner, output, 1)
