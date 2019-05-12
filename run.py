import datetime
import pandas as pd
from time import time
from src.kmeans import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances, adjusted_rand_score

party_map = {
        'Frente Amplio': 0,
        'Partido Nacional': 1,
        'Partido Colorado': 2,
        'La Alternativa': 3,
        'Unidad Popular': 4,
        'Partido de la Gente': 5,
        'PERI': 6,
        'Partido de los Trabajadores': 7,
        'Partido Digital': 8,
        'Partido Verde': 9,
        'Partido de Todos': 10
}

def load_data():
    # Importamos los datos utilizando pandas
    data=pd.read_csv("./data.csv")

    # Creo la tabla de candidatos a mano
    candidatos=pd.DataFrame(
        [
            [1,'Oscar Andrade', 'Frente Amplio'],
            [2,'Mario Bergara', 'Frente Amplio'],
            [3,'Carolina Cosse', 'Frente Amplio'],
            [4,'Daniel Martínez', 'Frente Amplio'],
            [5,'Verónica Alonso', 'Partido Nacional'],
            [6,'Enrique Antía', 'Partido Nacional'],
            [8,'Carlos Iafigliola', 'Partido Nacional'],
            [9,'Luis Lacalle Pou', 'Partido Nacional'],
            [10,'Jorge Larrañaga', 'Partido Nacional'],
            [11,'Juan Sartori', 'Partido Nacional'],
            [12,'José Amorín', 'Partido Colorado'],
            [13,'Pedro Etchegaray', 'Partido Colorado'],
            [14,'Edgardo Martínez', 'Partido Colorado'],
            [15,'Héctor Rovira', 'Partido Colorado'],
            [16,'Julio María Sanguinetti', 'Partido Colorado'],
            [17,'Ernesto Talvi', 'Partido Colorado'],
            [18,'Pablo Mieres', 'La Alternativa'],
            [19,'Gonzalo Abella', 'Unidad Popular'],        
            [20,'Edgardo Novick', 'Partido de la Gente'],
            [21,'Cèsar Vega', 'PERI'],
            [22,'Rafael Fernández', 'Partido de los Trabajadores'],
            [23,'Justin Graside', 'Partido Digital'],        
            [24,'Gustavo Salle', 'Partido Verde'],
            [25,'Carlos Techera', 'Partido de Todos']
        ],
        columns=['candidatoId','name','party'],
    )

    data=data.merge(candidatos,on=['candidatoId'])

    # Sólo por si necesita, cargamos un diccionario con el texto de cada pregunta
    preguntas={
        '1': 'Controlar la inflación es más importante que controlar el desempleo. ',
        '2': 'Hay que reducir la cantidad de funcionarios pùblicos',
        '3': 'Deberia aumentar la carga de impuestos para los ricos.',
        '4': 'El gobierno no debe proteger la industria nacional, si las fábricas no son competitivas esta bien que desaparezcan.',
        '5': 'La ley de inclusión financiera es positiva para la sociedad. ',
        '6': 'Algunos sindicatos tienen demasiado poder. ',
        '7': 'Cuanto más libre es el mercado, más libre es la gente. ',
        '8': 'El campo es y debe ser el motor productivo de Uruguay. ',
        '9': 'La inversión extranjera es vital para que Uruguay alcance el desarrollo. ',
        '10': 'Los supermercados abusan del pueblo con sus precios excesivos. ',
        '11': 'Con la vigilancia gubernamental (escuchas telefonicas, e-mails y camaras de seguridad) el que no tiene nada que esconder, no tiene de que preocuparse. ',
        '12': 'La pena de muerte debería ser una opción para los crímenes mas serios. ',
        '13': 'Uruguay debería aprobar más leyes anti corrupción y ser más duro con los culpables. ',
        '14': 'Las FF.AA. deberían tener un rol activo en la seguridad pública. ',
        '15': 'Las carceles deberían ser administradas por organizaciones privadas. ',
        '16': 'Hay que aumentar el salario de los policias significativamente. ',
        '17': 'Para los delitos más graves hay que bajar la edad de imputabilidad a 16 años. ',
        '18': 'Uruguay no necesita un ejército. ',
        '19': 'Uruguay es demasiado generoso con los inmigrantes. ',
        '20': 'La ley trans fue un error. ',
        '21': 'El feminismo moderno no busca la igualdad sino el poder. ',
        '22': 'Para la ley no deberia diferenciarse homicidio de femicidio. ',
        '23': 'La separación de estado y religión me parece importante. ',
        '24': 'La legalización de la marihuana fue un error. ',
        '25': 'La legalización del aborto fue un error. ',
        '26': 'El foco del próximo gobierno debe ser mejorar la educación pública. '
    }

    # Ordeno los datos por partido y luego por candidato

    data = data.sort_values(by=['party','name'])
    # Para PCA solo usamos las preguntas
    questions = data.iloc[:,2:28]
    return questions, data
    
def run():
    start_time = time()
    data, full_data = load_data()
    numeric_parties  = full_data.party.map(party_map)
    cluster_size = [2,3,5,10]
    for size in cluster_size:
        kmeans = KMeans(data)
        kmeans.train()
        result = metrics.silhouette_score(data, kmeans.data_cluster_idx, metric='euclidean')
        ari = adjusted_rand_score(numeric_parties, kmeans.data_cluster_idx)
        elapsed_time = time() - start_time
        print(f'----------------------------------------')
        print(f'###### CLUSTER SIZE: {size} ######')
        print(f'###### SILHOUETTE SCORE: {result} ######')
        print(f'###### ARI SCORE: {ari} ######')
        print(f'----------------------------------------')

    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')

if __name__ == "__main__":
    run()
