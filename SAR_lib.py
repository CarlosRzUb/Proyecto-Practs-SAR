# versión 1.1

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel
from nltk.tokenize import sent_tokenize


# INICIO CAMBIO EN v1.1
## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()
# FIN CAMBIO EN v1.1


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            # INICIO CAMBIO EN v1.1
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)
            # FIN CAMBIO EN v1.1

            
            

    # INICIO CAMBIO EN v1.2

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        sentences = sent_tokenize(txt)  # Se divide el texto en frases individuales
        self.chuncks.extend(sentences)  # Añadimos las frases extraídas a la lista general de chunks

        # Codificamos las frases con el modelo de embeddings
        start_idx = len(self.embeddings)  # Índice desde el que empezaremos a insertar los nuevos embeddings
        emb = self.model.encode(sentences)  # Codificamos las frases en vectores semánticos
        self.embeddings.extend(emb)  # Añadimos los vectores resultantes a la lista general

        # Actualizamos la estructura que asocia cada chunk a su artículo y su posición en dicho artículo
        for i in range(len(sentences)):
            self.chunck_index.append((artid, i))  # Añadimos una tupla con el identificador del artículo y la posición del chunk

        # Asociamos el artículo con los índices de los embeddings recién añadidos
        self.artid_to_emb.setdefault(artid, []).extend(range(start_idx, start_idx + len(sentences)))
        '''
        setdefault sirve para inicializar una clave si no existe:
        - Si artid ya existe como clave, usamos directamente su lista asociada
        - Si no existe, se crea una lista vacía y se añaden los índices de los nuevos embeddings
        '''             
        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
        self.kdtree = self.model.fit(self.embeddings) # Con el .fit() se crea el kdtree a partir de los embeddings
        print("done!")


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()  # Aseguramos que el modelo semántico está cargado

        top_k = self.MAX_EMBEDDINGS  # Empezamos buscando un número limitado de resultados
        retrieved_chunks = []  # Lista de chunks que coinciden con la consulta
        retrieved_articles = set()  # Conjunto de artículos extraídos a partir de los chunks

        while True:
            results = self.model.query(query, top_k)  # Obtenemos los top_k chunks más cercanos a la query
            retrieved_chunks = results  # Guardamos los resultados como lista de tuplas (chunk_idx, distancia)

            if not retrieved_chunks:  # Si no se recupera nada, salimos
                break

            max_dist = retrieved_chunks[-1][1]  # Distancia del último chunk recuperado
            new_articles = {self.chunck_index[i][0] for i, _ in retrieved_chunks}  # Convertimos los chunks en sus artículos correspondientes
            retrieved_articles = new_articles

            # Si ya hemos alcanzado el umbral o todos los embeddings, terminamos la búsqueda
            if max_dist > self.semantic_threshold or len(retrieved_chunks) >= len(self.embeddings):
                break

            # Si no, aumentamos top_k y seguimos recuperando
            top_k = min(len(self.embeddings), top_k + self.MAX_EMBEDDINGS)

        return list(retrieved_articles)  # Devolvemos la lista de artículos relevantes


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()  # Aseguramos que el modelo esté cargado

        top_k = self.MAX_EMBEDDINGS  # Comenzamos buscando una cantidad limitada
        article_set = set(articles)  # Convertimos la lista original en conjunto para comparación rápida

        while True:
            results = self.model.query(query, top_k)  # Consultamos el árbol semántico
            chunk_idxs = [i for i, _ in results]  # Extraemos los índices de los chunks recuperados
            retrieved_articles = [self.chunck_index[i][0] for i in chunk_idxs]  # Obtenemos los artículos asociados a cada chunk

            if article_set.issubset(set(retrieved_articles)) or top_k >= len(self.embeddings):
                # Si ya hemos recuperado todos los artículos relevantes o no se puede aumentar más, salimos
                break

            # Si aún faltan artículos, aumentamos el número de resultados a recuperar
            top_k = min(len(self.embeddings), top_k + self.MAX_EMBEDDINGS)

        # Creamos una lista ordenada de artículos según el primer chunk que aparece en los resultados
        seen = set()  # Para evitar añadir artículos repetidos
        reranked = []  # Lista final de artículos reordenados

        for i, _ in results:
            aid = self.chunck_index[i][0]  # Obtenemos el artículo al que pertenece el chunk
            if aid in article_set and aid not in seen:  # Solo añadimos si pertenece al conjunto y no se ha añadido antes
                reranked.append(aid)
                seen.add(aid)
            if len(reranked) == len(articles):  # Paramos si ya los hemos añadido todos
                break

        return reranked  # Devolvemos la lista reordenada
    
    # FIN CAMBIO EN v1.2

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    ###############################
    ###                         ###
    ###     JAIME Y NACHO       ###
    ###                         ###
    ###############################

    # Método auxiliar ya hecho
    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls

    # Método auxiliar ya hecho
    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        
    # Método auxiliar ya hecho    
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article

    # Método para indexar que tenemos que completar
    def index_file(self, filename: str):
        '''
        El objetivo de index_file es crear un índice para todos los documentos JSON que a su
        vez contienen artículos dentro.
        
        Es fácil observar que por cada documento tendremos
        que asignar un entero en self.docs y luego por cada lína desde la 1 hasta la n
        tendremos que asignar otro entero al artículo en self.articles.
        
        Por tanto lo que vamos a hacer es recorrer estos documentos e ir asignando pares
        clave valor al self; por ejemplo, para el documento 1 línea 7 tendremos como clave 
        docid = 1 y artid = 7 y así podremos diferenciarlo de los demás.
        
        Además para cada aparición de un token tendremos un clave valor asociado donde
        la clave será el artículo asociado y el valor una lista de [apariciones del token] ---> Búsqueda posicional
        en ese documento en las posiciones relativas.
        '''
        # Accedemos a los valores de self.docs para ver si ya ha sido procesado
        if filename not in self.docs.values():
            docid = len(self.docs)  # Asignamos valor con len para que sea incremental en orden
            self.docs[docid] = filename # Guardamos en self.docs con clave docid
        else:
            # Si el fichero ya ha sido procesado seguimos
            return

        docid = list(self.docs.keys())[-1]  # Cogemos el último docid asignado

        # Reccoremos cada línea del JSON
        for i, line in enumerate(open(filename, encoding="utf-8")): # Usamos enumerate para saber el número de línea 0...n y line el contenido de esta
            article = self.parse_article(line)
            '''
            if self.already_in_index(article):
                continue    # Saltamos los artículos ya indexados con el método auxiliar
            '''
            artid = len(self.articles)  # Asignamos valor con len para que sea incremental en orden
            self.articles[artid] = (docid, i)  # i = posición del artículo en el fichero
            self.urls.add(article['url'])  # Marcamos la URL como indexada

            # PONGO ALL PORQUE CON DEFAULT_FIELD nos está dando problemas
            tokens = self.tokenize(article['all'])  # Limpiamos el artículo con tokenize

            # Ahora es turno de crear el índice invertido
            positions = {}
            for pos, token in enumerate(tokens):    # Volvemos a usar enumerate para lo mismo que con line arriba
                if not token: # Si token es vacío ignoramos, PUEDE QUE NO HAGA FALTA, HAY QUE COMPROBAR
                    continue

                if token not in self.index: # Si token no está inicializamos lista vacía
                    self.index[token] = []

                # Si se activa el índice posicional
                if self.positional:
                    # Hay que guardar cada artid para el token
                    if self.index[token] and self.index[token][-1][0] == artid: # Si en ESTE MISMO ARTID YA EXISTE EL TOKEN INDEXADO solo guardamos la nueva posición donde aparece
                        self.index[token][-1][1].append(pos)
                    else:   # Si se trata de OTRO ARTID tenemos que poner otra NUEVA ENTRADA para el token y su posición
                        self.index[token].append((artid, [pos]))
                        
                # Si el índice no es posiconal        
                else:
                    # Solo guardamos el artid una vez por término
                    if not any(entry[0] == artid for entry in self.index[token]):   #Recorremos cada entrada asociada al token para saber si se corresponde con el artid actual
                        self.index[token].append((artid, []))  # Lista vacía de posiciones
        '''
        Si no activamos positional con -P podremos buscar si un término aparece en un artículo
        y con esto hacer búsquedas de tipo OR o AND.
        
        Sin embargo si activamos positional con -P guardaremos la posición relativa de cada 
        término en cada artículo; esto nos permitirá hacer búsquedas más avanzadas: frases
        exactas o proximidad. Esta es mucho más interesante y útil como se puede ver.
        '''
        

    # Método auxiliar ya hecho
    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()

    # Método para mostrar estadísticas que tenemos que completar
    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        # No tiene más misterio que acceder a los atributos del self donde hayamos guardado las estadísticas y mostrarlas con f strings
        print("=" * 40)
        print(f"Number of indexed files: {len(self.docs)}")
        print("-" * 40)
        print(f"Number of indexed articles: {len(self.articles)}")
        print("-" * 40)
        print("TOKENS:")
        print(f"\t# of tokens in 'all': {len(self.index)}")
        print("-" * 40)

        if hasattr(self, 'positional') and self.positional:
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")

        print("=" * 40)



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################
    
    #################################
    ###                           ###
    ###      Carlos y Adrián     ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        
        if not query:
            return [], []

        query = query.strip()
        used_terms = []

        # 1. Buscar frases exactas (búsqueda posicional)
        if '"' in query:
            # Extrae lo que está entre comillas
            parts = re.findall(r'"(.*?)"', query)
            if parts:
                phrase = parts[0]
                terms = self.tokenize(phrase)
                used_terms.extend(terms)
                result = self.get_positionals(terms)
                return [artid for artid, _ in result], used_terms

        # 2. Tokenización de la consulta normal y detección de operadores
        tokens = query.split()
        terms = []
        operators = []

        i = 0
        while i < len(tokens):
            token = tokens[i].upper()
            if token in ['AND', 'OR', 'NOT']:
                operators.append(token)
            else:
                term = tokens[i].lower()
                posting = [artid for artid, _ in self.get_posting(term)]
                used_terms.append(term)
                terms.append(posting)
            i += 1

        if not terms:
            return [], []

        # 3. Procesamiento de operadores lógicos (de izquierda a derecha)
        result = terms[0]
        op_idx = 0

        for i in range(1, len(terms)):
            op = operators[op_idx] if op_idx < len(operators) else 'AND'
            next_posting = terms[i]

            if op == 'AND':
                result = self.and_posting(result, next_posting)
            elif op == 'OR':
                result = sorted(list(set(result).union(set(next_posting))))
            elif op == 'NOT':
                result = self.and_posting(result, self.reverse_posting(next_posting))

            op_idx += 1

        return result, used_terms

    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        # Si el término no está en el índice, devolvemos una lista vacía
        if term not in self.index:
            return []
        return self.index[term]

    def get_positionals(self, terms:str):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """
        
        # Paso 1: obtener posting lists de todos los términos
        result = []
        for term in terms:
            if term not in self.index:
                return []
            result.append(self.index[term])

        # Paso 2: hacer intersecciones posicionales entre cada par consecutivo
        posting = result[0]
        for i in range(1, len(result)):
            p1 = posting
            p2 = result[i]
            new_posting = []
            i1 = i2 = 0
            
            # Se recorre la posting list de ambos términos
            while i1 < len(p1) and i2 < len(p2):
                doc1, pos1 = p1[i1]
                doc2, pos2 = p2[i2]
                if doc1 == doc2:
                    matches = []
                    idx1 = 0
                    idx2 = 0
                    while idx1 < len(pos1):
                        while idx2 < len(pos2):
                            if pos2[idx2] - pos1[idx1] == 1:
                                matches.append(pos2[idx2])
                                break  # una coincidencia por pos1 es suficiente
                            elif pos2[idx2] > pos1[idx1] + 1:
                                break
                            idx2 += 1
                        idx1 += 1
                    if matches:
                        new_posting.append((doc1, matches))
                    i1 += 1
                    i2 += 1
                elif doc1 < doc2:
                    i1 += 1
                else:
                    i2 += 1
            posting = new_posting
            if not posting:
                return []

        return posting
        
    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        result = []
        for i in range(len(self.articles)):
            if i not in p:
                result.append(i)
        return result

    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        
        result = []

        i, j = 0, 0
        while i <= (len(p1) - 1) and j <= (len(p2) - 1):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        
        return result

    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################

    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        if not query:
            return 0

        result, _ = self.solve_query(query)

        print(f'{query}\t{len(result)}')
        if result:
            self.show_stats()

        return len(result)