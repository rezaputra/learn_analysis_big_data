from mrjob.step import MRStep

class GenderCount(MRJob):
  def steps(self):
      return [
          MRStep(mapper=self.mapper1,
                  reducer=self.reducer1),
          MRStep(mapper = self.mapper2,
                  reducer=self.reducer2)
      ]
  def mapper1(self, _, row):
    yield ((row['gender'], row['start_station_name']), 1)

  def reducer1(self, gender_station, count):
    yield (gender_station, sum(count))

  def mapper2(self, gender_station, count):
    gender, station  = gender_station
    yield (gender, (station, count))

  def reducer2(self, gender, station_count):
    genderMap = {'0':'Unknown', '1':'Male', '2':'Female'}
    yield (genderMap[gender], max(station_count, key = lambda x : x[1]))


task1 = GenderCount(args = [])
with open('main_data.csv', 'r') as fi:
  output = list(mr.runJob(enumerate(csv.DictReader(fi)), task1))