from ComputeEngineManagerMTCMPS import ComputeEngineManager, ComputeEngine
from kafka import KafkaConsumer
import ComputeEngineService.constants as constants

class ComputeEngineConsumer():
    
    def __init__(self, compute_engines, topic_name):
        self.compute_engine_manager = ComputeEngineManager(compute_engines)
        self.cosumer = KafkaConsumer(client_id = constants.client_id,
                                     group_id = constants.group_id,
                                     bootstrap_service_hostname=constants.bootstrap_service_hostname
                                     bootstrap_service_port=constants.bootstrap_service_port, 
                                     max_poll_records=constants.max_poll_records)
        self.consumer.subscribe(topic_name)
        self.requests = []
        self.results = []
        
    def run(self):
        t = Thread(target=self.answer)
        t.start()
        t = Thread(target=self.retrieve)
        t.start()
        
    def answer(self):
        
        for message in self.consumer:
            thread, result = self.compute_engine_manager.request(message, block=False)
            self.requests += [ (thread, result) ]
            
    def retrieve(self):
        
        while True:
            for thread, result in self.requests:
                if result[0] is not None:
                    self.results += result
                
            
        