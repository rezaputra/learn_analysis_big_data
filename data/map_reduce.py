from mrjob.step import MRStep

class DrugCount(MRJob):
  def steps(self):
      return [
          MRStep(mapper=self.mapper1,
                  reducer=self.reducer1),
          MRStep(mapper = self.mapper2,
                  reducer=self.reducer2)
      ]
  def mapper1(self, _, row):
    yield ((row['Drug'], row['Na_to_k']), 1)

  def reducer1(self, Na_to_k, count):
    yield (Na_to_k, sum(count))

  def mapper2(self, Na_to_k, count):
    gender, station  = Na_to_k
    yield (Drug, (station, count))

  def reducer2(self, Drug, station_count):
    genderMap = {'0':'Unknown', '1':'Male', '2':'Female'}
    yield (genderMap[gender], max(station_count, key = lambda x : x[1]))


task1 = DrugCount(args = [])
with open('main_data.csv', 'r') as fi:
  output = list(mr.runJob(enumerate(csv.DictReader(fi)), task1))