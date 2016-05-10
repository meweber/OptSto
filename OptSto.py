import functools
import matplotlib.pyplot as plt
import numpy as np
import operator
from random import randint


#MODEL
groupmembers = 18
teamnameV = np.matrix(["Toulouse", "Cergy", "Meudon", "Amneville", "Francais_Volants", "Asnieres", "Valence", "Avignon", "Marseille", "Chambery", "Annecy", "Limoges", "Clermont", "Villard", "Roanne", "Evry", "Strasbourg", "Wasquehal"])
N = len(teamnameV)
distanceM = np.matrix('0.0 717.0 667.0 1015.0 673.0 686.0 449.0 344.0 417.0 604.0 665.0 291.0 408.0 517.0 504.0 668.0 973.0 901.0; 717.0 0.0 52.0 364.0 42.0 22.0 599.0 727.0 813.0 610.0 597.0 426.0 458.0 634.0 435.0 70.0 526.0 236.0; 667.0 52.0 0.0 340.0 7.0 20.0 558.0 686.0 772.0 569.0 556.0 376.0 409.0 593.0 394.0 29.0 502.0 239.0; 1015.0 364.0 340.0 0.0 332.0 336.0 572.0 700.0 787.0 583.0 555.0 725.0 615.0 607.0 487.0 340.0 176.0 391.0; 673.0 42.0 7.0 332.0 0.0 14.0 563.0 691.0 777.0 574.0 561.0 385.0 417.0 598.0 399.0 35.0 494.0 232.0; 686.0 22.0 20.0 336.0 14.0 0.0 574.0 702.0 788.0 585.0 572.0 396.0 428.0 609.0 410.0 46.0 507.0 224.0; 449.0 599.0 558.0 572.0 563.0 574.0 0.0 130.0 217.0 154.0 216.0 506.0 264.0 68.0 205.0 531.0 592.0 793.0; 344.0 727.0 686.0 700.0 691.0 702.0 130.0 0.0 95.0 286.0 347.0 634.0 392.0 199.0 333.0 659.0 720.0 921.0; 417.0 813.0 772.0 787.0 777.0 788.0 217.0 95.0 0.0 372.0 434.0 692.0 479.0 286.0 419.0 746.0 807.0 1007.0; 604.0 610.0 569.0 583.0 574.0 585.0 154.0 286.0 372.0 0.0 62.0 516.0 273.0 102.0 197.0 542.0 479.0 804.0; 665.0 597.0 556.0 555.0 561.0 572.0 216.0 347.0 434.0 62.0 0.0 558.0 316.0 161.0 240.0 530.0 432.0 775.0; 291.0 426.0 376.0 725.0 385.0 396.0 506.0 634.0 692.0 516.0 558.0 0.0 259.0 540.0 289.0 381.0 731.0 613.0; 408.0 458.0 409.0 615.0 417.0 428.0 264.0 392.0 479.0 273.0 316.0 259.0 0.0 296.0 111.0 410.0 635.0 642.0; 517.0 634.0 593.0 607.0 598.0 609.0 68.0 199.0 286.0 102.0 161.0 540.0 296.0 0.0 219.0 564.0 597.0 826.0; 504.0 435.0 394.0 487.0 399.0 410.0 205.0 333.0 419.0 197.0 240.0 289.0 111.0 219.0 0.0 370.0 505.0 622.0; 668.0 70.0 29.0 340.0 35.0 46.0 531.0 659.0 746.0 542.0 530.0 381.0 410.0 564.0 370.0 0.0 503.0 255.0; 973.0 526.0 502.0 176.0 494.0 507.0 592.0 720.0 807.0 479.0 432.0 731.0 635.0 597.0 505.0 503.0 0.0 553.0; 901.0 236.0 239.0 391.0 232.0 224.0 793.0 921.0 1007.0 804.0 775.0 613.0 642.0 826.0 622.0 255.0 553.0 0.0')
standingV = np.matrix('1 3 4 7 8 11 12 16 18 2 5 6 9 10 13 14 15 17')

class Individuum(object):

	def __init__(self, DNA):
		self.DNA = DNA
		self.totalDistance = -1
		self.totalStanding = -1
		self.eval()

# TODO :: check correctness of the function
	def dominates (self, other):
		return (self.totalDistance <= other.totalDistance and self.totalStanding <= other.totalStanding)
		
	#EVALUTAION
	def eval(self):
		self.evalDistance()
		self.evalStanding() 

# TODO :: check correctness of the function
	def evalDistance(self):
		self.totalDistance = (self.DNA * distanceM * self.DNA.T)[0,0]

# TODO :: check correctness of the function
	def evalStanding(self):
		# 1. Version
		self.totalStanding = np.fabs((self.DNA * standingV.T) - (np.logical_not(self.DNA) * standingV.T))[0,0]
		# 2. Version
		self.totalStanding = self.calculateTotalStanding()
		
	def calculateTotalStanding(self):
	# ToDo
	
#TODO
	#COMPARING two individuums
	def __lt__(self, other):
		return self.totalDistance < other.totalDistance and self.totalStanding < other.totalStanding
	
	def __le__(self, other):
		#if not dominated by other
		return not(self.totalDistance > other.totalDistance and self.totalStanding > other.totalStanding)

	def __gt__(self, other):
		return self.totalDistance > other.totalDistance and self.totalStanding > other.totalStanding

	def __ge__(self, other):
		return not(self.totalDistance < other.totalDistance and self.totalStanding < other.totalStanding)

	def __eq__(self, other):
		return (self.totalDistance == other.totalDistance and self.totalStanding == other.totalStanding)

class Generation(object):
	
	def __init__(self, genLevel, individuums):
		self.genLevel = genLevel
		self.name = "Generation "+str(genLevel)
		self.individuums = individuums
		
		# all the individuums, that aren't dominated by any other individuum (prank == 0)
		self.paretoFront = []
		
		# Matrix of the form:
		#  [0] all the individuums with prank == 0
		#  [1] all the individuums with prank == 1
		#  ...
		self.fronts = []		# fronts[0][0] = (individuum, dominationCount, dominatedIndividuums)
		
		self.totalDistance = -1
		self.totalStanding = -1
		self.distances = []
		self.standings = []
		
		self.eval()


	def eval(self):
		self.evalDistance()
		self.evalStanding()
		
	def evalDistance(self):
		totalDist = 0
		self.distances = []
		
		for indiv in self.individuums:
			distance = indiv.totalDistance
			totalDist += distance
			self.distances.append(distance)
		
		self.totalDistance = totalDist

	def evalStanding(self):
		totalStand = 0
		self.standings = []
		
		for indiv in self.individuums:
			standing = indiv.totalStanding
			totalStand += standing
			self.standings.append(standing)
		
		self.totalStanding = totalStand

	def sortGen(self):
		sortedIndivs = []
		indivs = self.individuums.copy()
		
		while len(indivs)>0:
			# minIndiv: one element, that is not dominated by any of the others
			minIndiv = min(indivs)
			sortedIndivs.append(minIndiv)
			indivs.remove(minIndiv)
		
		self.individuums = sortedIndivs
		self.eval()

class Evolution(object):

	def __init__(self, generations):
		self.generations = generations
		self.paretoFront = []

class Archive(object):
	
	def __init__(self, size):
		self.PF1 = []
		self.size = size #TODO


#VIEW
class View(object):
	def __init__(self):
		i = 0
		
	def draw1Generation(self, gen, limit):
		plt.title(gen.name)
		self.diaProps()
		plt.plot(np.array(gen.distances)[0:limit], np.array(gen.standings)[0:limit], 'go')
		plt.plot(np.array(gen.distances)[limit:], np.array(gen.standings)[limit:], 'ro')
		plt.show()
	
	def drawNGenerations(self, nGen, n):
		self.diaProps()
		plt.title("Start Population and some Generations")
		
		i = 0
		if (len(n)>i and len(nGen)>n[i]):
			plt.plot(np.array(nGen[n[i]].distances), np.array(nGen[n[i]].standings), 'yo', label=nGen[n[i]].name)
		
		i = 1
		if (len(n)>i and len(nGen)>n[i]):
			plt.plot(np.array(nGen[n[i]].distances), np.array(nGen[n[i]].standings), 'go', label=nGen[n[i]].name)
		
		i = 2
		if (len(n)>i and len(nGen)>n[i]):
			plt.plot(np.array(nGen[n[i]].distances), np.array(nGen[n[i]].standings), 'bo', label=nGen[n[i]].name)
		
		i = 3
		if (len(n)>3 and len(nGen)>n[i]):
			plt.plot(np.array(nGen[n[i]].distances), np.array(nGen[n[i]].standings), 'ro', label=nGen[n[i]].name)
		
		plt.legend(numpoints=1)
		plt.show()
	
	def drawParetofront(self, paretofront):
		self.diaProps()
		plt.title("Pareto-front")
		
		xs = []
		ys = []
		for i in paretofront:
			xs.append(i.totalDistance)
			ys.append(i.totalStanding)
		plt.plot(xs, ys, 'ro', label='archive data (prank==0)')
		plt.show()
	
	def diaProps(self):
		plt.xlabel('total distances')
		plt.ylabel('standing difference')
		plt.grid(True)
	
	def info(self):
		print()
		print("+------------------------------------------------------------+")
		print("| Project n°3 : 'Composition des poules pour un championnat' |")
		print("+------------------------------------------------------------+")
		print("|      author : Meike WEBER                                  |")
		print("|     version : 0.4, 2016-05-07, 22:00                       |")		
		print("+------------------------------------------------------------+")
		print()

		
#CONTROLLER
class Controler(object):
	def __init__(self):		
		self.evolutions = []
		self.view = View()


	def userInput(self):
		self.view.info()
		self.numI           = int(input("  Number of individuums per generation: "))
		self.numG           = int(input("  Number of generations:                "))
		self.perInh = 1.0 - float(input("  Death rate:                           "))
		print()
	
	def autoInput(self, numI, numG, perDeath):
		self.view.info()
		print("  Number of individuums per generation: "+str(numI))
		self.numI = numI
		print("  Number of generations:                "+str(numG))
		self.numG = numG
		print("  Death rate:                           "+str(perDeath))
		self.perInh = 1-perDeath
		print()


	def visualiseEvolution(self, gen, n):
		self.view.drawNGenerations(gen, n)

	def visualiseResult(self, paretoFront):
		self.view.drawParetofront(paretoFront)

#CREATE
	def createEvolution(self):
		self.limit = int(self.numI*self.perInh)
		generations = []
	#FstGen
		fstGeneration = self.createStartPopulation(self.numI)
		fstGeneration.sortGen()
		generations.append(fstGeneration)
#old	self.archive.PF1.append(fstGeneration.paretoFront)
		
	#iGen
		lastGen = fstGeneration
		for i in range(self.numG):
			nextGen = self.createNextGeneration(lastGen, self.numI)
			nextGen.sortGen()
			generations.append(nextGen)
#old		self.archive.PF1.append(nextGen.paretoFront)
			lastGen = nextGen
			
		e = Evolution(generations)
		e.paretoFront = generations[len(generations)-1].paretoFront
		
		return e

	def createNEvolutions(self, n):
		self.evolutions = []
		fronts = []
		
		for x in range(n):
			e = self.createEvolution()
			fronts += e.paretoFront
			self.evolutions.append(e)
		
		paretoFront = self.calcParetoFront_Kung75(fronts)
		
		return (self.evolutions, paretoFront)

	
	def createStartPopulation(self, numIndividuums):
		individuums = []
		for x in range(numIndividuums):
			individuums.append(self.createRandomIndividuum())
			
		newG = Generation(0, individuums)
		#newG.fronts = self.calcAllFronts(individuums)
		newG.paretoFront = self.calcParetoFront_Kung75(individuums)
		
		return newG
		
	def createRandomIndividuum(self):
		#array PFilled with 'True'
		DNA = np.ones(groupmembers, dtype=bool)
		#positions for false in array
		falses = np.random.permutation(groupmembers)[0:int(groupmembers/2)]
		DNA[falses]=False
		return Individuum(np.matrix(DNA.reshape((1,groupmembers))))


	def createNextGeneration(self, lastGen, numIndividuums):
	#SELECTION (of parents)
		parents = lastGen.individuums[0:self.limit]
		
		# retake the pareto-optimal individuums of the previous generation
		paretos = self.calcParetoFront_Kung75(lastGen.individuums)
		newIs = paretos
		
		# cross all the parto-individuums with each other
		for m in range(len(paretos)):
			for f in range(len(paretos)):
				if m != f:
					newIs.append(self.createChild(paretos[m], paretos[f]))

		# create 'numIndividuums' childs
		for x in range(numIndividuums - len(newIs)):
			motherIndex = randint(0,self.limit-1)
			fatherIndex = randint(0,self.limit-1)
			while(motherIndex==fatherIndex):
				fatherIndex = randint(0,self.limit-1)
			
			mother = parents[motherIndex]
			father = parents[fatherIndex]

			newIs.append(self.createChild(mother, father))
		
		newG = Generation((lastGen.genLevel+1), newIs)
#old	newG.fronts = self.calcAllFronts(individuums)
		newG.paretoFront = self.calcParetoFront_Kung75(newIs)
		
		return newG

	def createChild(self, mother, father):
		motherDNA = mother.DNA
		fatherDNA = father.DNA

	#INHERITENCE
		childDNA = motherDNA
	
	#MUTATION
		# positions in DNA where motherDNA.value != fatherDNA.value
		mismatchIndex = (motherDNA != fatherDNA)
		numMismatch = np.sum(mismatchIndex)
		
		# number of 'True'/'False' in motherDNA in mismatch
		numTrue = np.sum(motherDNA[mismatchIndex])
		numFalse = numMismatch - numTrue
		
		# random index for 'False' for 'mismatchIndex'
		falses = np.random.permutation(numMismatch)[0:numFalse]
		
		# set 'False' for all mismatchIndex-positions in childDNA
		childDNA[mismatchIndex]=False
		
		# only 'True' for indexes, that are not falses
		trues = -1
		for x in range(np.shape(mismatchIndex)[1]):
			if(mismatchIndex[0,x]):
				trues += 1
				if(trues in falses):
					mismatchIndex[0,x]=False
		
		# set 'True' for random true-index
		childDNA[mismatchIndex]=True
		
		return Individuum(childDNA)

#CALCULATE
	#for calculating the rangs (NSGA2)
	def calcAllFronts(self, P):
		# Matrix of the form:
		#  [0] all the I with prank == 0
		#  [1] all the I with prank == 1
		#  ...
		F = [[]]

		PF1 = []		# all individuums, that aren't dominated by any other individuum
		PFN = []		# all individuums, that are dominated by the previous front
		
		# calculate 'first' front
		x = self.calcParetoFront(P)
		F[0] = x[0]
		PFN = x[1]
		
		# calculate 'next' fronts [different from PAPER(Dep,2002)]
		while (True):
			PFi = []
			R = []		# = rest
			for p in PFN:
				np = 0
				for q in PFN:
					if (p[0].dominates(q[0])):
						R.append(q)
						PFN.remove(q)
					else:
						if (q[0].dominates(p[0])):
							np += 1
				if (np == 0):
					PFi.append(p)
				else:
					R.append(p)
			F.append(PFi)
			if (not R):
				break
			PFN = R
		
		return F

	def calcParetoFront(self, P):
		PF1 = []		# all individuums, that aren't dominated by any other individuum (prank == 0)
		PFN = []		# all individuums, that are dominated by the 'first' front (prank > 0)
		
		# calculate First front
		for p in P:
			Sp = []		# all individuums that are dominated by p
			np = 0		# by how many individuums is p dominated?
			
			for q in P:
				if (p.dominates(q)):
					Sp.append(q)
				else:
					if (q.dominates(p)):
						np += 1
			
			if (np == 0):
				PF1.append((p, np, Sp))
			else:
				PFN.append((p, np, Sp)) # [different from PAPER(Dep,2002)]
		
		return (PF1, PFN)

	def calcParetoFront_Kung75(self, individuums):
		# sort Individuums concerning one criteria (Distance)
		sortedI = sorted(individuums, key=lambda individuum: individuum.totalDistance)
		
		# calculate front by "devide and conquer"-approach
		paretoI = self.recursiveKung75(sortedI)
		
		return(paretoI)
		
	def recursiveKung75(self, sortedI):
		# DEVIDE
		n = len(sortedI)
		
		if(n >= 2):
			# DEVIDE			
			paretoI = self.recursiveKung75(sortedI[0:int(n/2)])
			optimaI = self.recursiveKung75(sortedI[int(n/2):n])
							
			# CONQUER
			for i in optimaI:
				paretoOptimal = True
				
				for pi in paretoI:
					if i.totalStanding >= pi.totalStanding:
						paretoOptimal = False
						break
						
				if paretoOptimal:
					paretoI.append(i)
			
			return paretoI
			
		else:
			# CONQUER (n==0)
			return sortedI


#MAIN
c = Controler()
#c.userInput()
c.autoInput(100,10,0.5)
e = c.createNEvolutions(10)

c.visualiseResult(e[1])

for x in e[0]:
	c.visualiseEvolution(x.generations, [0,2,4,6])