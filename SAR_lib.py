# versión 1.1

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import KDTree, SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel
from nltk.tokenize import sent_tokenize
import numpy as np


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
                  "semantic", "chunks", "embeddings", "chunk_index", "kdtree", "artid_to_emb"]


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
        self.chunks = [] # Lista de chunks
        self.embeddings = [] # Lista de embeddings de los chunks
        self.chunk_index = [] # Lista de indices de los chunks
        self.artid_to_emb = {} # Diccionario de clave: artid, valor: media de los embeddings de los chunks del articulo
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta # número máximo de embedding que se extraen del kdtree en una consulta
        
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
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
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
            print(f"loading {modelname} model ... ", end="", file=sys.stderr)
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)
            # Añadimos esto para restaurar el kdtree y embeddings si existen
            if hasattr(self, "kdtree") and self.kdtree is not None and hasattr(self.model, "set_kdtree"):
                self.model.set_kdtree(self.kdtree)
            if hasattr(self, "embeddings") and self.embeddings is not None and hasattr(self.model, "set_embeddings"):
                self.model.set_embeddings(np.array(self.embeddings))     

    # INICIO CAMBIO EN v1.2

    def update_chunks(self, txt: str, artid: int):
        # Asegura que el modelo semántico esté cargado antes de procesar los textos
        self.load_semantic_model()
        
        # Divide el texto completo del artículo en frases (sentencias)
        sentences = sent_tokenize(txt)
        # Añade las nuevas frases al listado global de chunks
        self.chunks.extend(sentences)

        # Guarda el índice inicial donde se empezarán a añadir los nuevos embeddings
        start_idx = len(self.embeddings)
        # Calcula los embeddings para cada frase usando el modelo semántico
        emb = self.model.get_embeddings(sentences)
        # Añade los nuevos embeddings al listado global
        self.embeddings.extend(emb)

        # Por cada frase añadida, guarda una tupla (artid, i) en chunk_index,
        # donde 'artid' es el identificador del artículo y 'i' el índice de la frase dentro del artículo
        for i in range(len(sentences)):
            self.chunk_index.append((artid, i))

        # Actualiza el diccionario artid_to_emb para este artículo,
        # añadiendo los índices de los nuevos embeddings asociados a este artid
        self.artid_to_emb.setdefault(artid, []).extend(range(start_idx, start_idx + len(sentences)))



    def create_kdtree(self):
        print(f"Creating kdtree ...", end="")
        
        # Convierte la lista de embeddings a un array de numpy si es necesario
        if isinstance(self.embeddings, list):
            embeddings_array = np.array(self.embeddings)
        else:
            embeddings_array = self.embeddings

        # Crea el KDTree usando los embeddings y la métrica euclídea
        self.kdtree = KDTree(embeddings_array, metric="euclidean")
        
        # Si el modelo tiene el método set_kdtree, le pasa el KDTree creado
        if hasattr(self.model, "set_kdtree"):
            self.model.set_kdtree(self.kdtree)
        
        print("done!")



    def solve_semantic_query(self, query: str):
        # Convierte la consulta a minúsculas para normalizarla
        query = query.lower()

        # Asegura que el modelo semántico esté cargado antes de realizar la consulta
        self.load_semantic_model()
        top_k = self.MAX_EMBEDDINGS  # Número inicial de embeddings a recuperar
        retrieved_articles = set()   # Conjunto para almacenar los artículos recuperados

        while True:
            # Realiza la consulta semántica al modelo, recuperando los top_k resultados más similares
            results = self.model.query(query, top_k)
            if not results:
                break  # Si no hay resultados, termina el bucle

            # Si hay un umbral de similitud definido, filtra los resultados por distancia
            if self.semantic_threshold is not None:
                filtered = [(dist, ind) for dist, ind in results if dist <= self.semantic_threshold]
            else:
                filtered = results

            if not filtered:
                break  # Si no quedan resultados tras el filtrado, termina el bucle

            # Obtiene los identificadores de artículos asociados a los chunks recuperados
            new_articles = {self.chunk_index[ind][0] for dist, ind in filtered}
            retrieved_articles = new_articles

            # Si se han recuperado menos de top_k resultados o ya se han explorado todos los embeddings, termina
            if len(filtered) < top_k or len(filtered) >= len(self.embeddings):
                break

            # Aumenta el número de resultados a recuperar en la siguiente iteración (si es necesario)
            top_k = min(len(self.embeddings), top_k + self.MAX_EMBEDDINGS)

        # Devuelve la lista de identificadores de artículos recuperados
        return list(retrieved_articles)


    '''
    def semantic_reranking(self, query: str, articles: List[int]):
        self.load_semantic_model()

        terms = [t for t in self.tokenize(query) if t]

        sets = []
        for term in terms:
            top_k = self.MAX_EMBEDDINGS
            retrieved_articles = set()
            while True:
                results = self.model.query(term, top_k)
                if not results:
                    break
                if self.semantic_threshold is not None:
                    filtered = [(dist, ind) for dist, ind in results if dist <= self.semantic_threshold]
                else:
                    filtered = results
                if not filtered:
                    break
                new_articles = {self.chunk_index[ind][0] for dist, ind in filtered}
                retrieved_articles.update(new_articles)
                if len(filtered) < top_k or len(filtered) >= len(self.embeddings):
                    break
                top_k = min(len(self.embeddings), top_k + self.MAX_EMBEDDINGS)
            sets.append(retrieved_articles)

        if sets:
            article_set = set.intersection(*sets)
        else:
            article_set = set()

        top_k = self.MAX_EMBEDDINGS
        reranked = []
        seen = set()
        while True:
            results = self.model.query(query, top_k)
            for _, ind in results:
                aid = self.chunk_index[ind][0]
                if aid in article_set and aid not in seen:
                    reranked.append(aid)
                    seen.add(aid)
                if len(reranked) == len(article_set):
                    break
            if len(reranked) == len(article_set) or top_k >= len(self.embeddings):
                break
            top_k = min(len(self.embeddings), top_k + self.MAX_EMBEDDINGS)

        return reranked
    '''
    
    # FIN CAMBIO EN v1.2

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls



    def index_dir(self, root: str, **args):
        """
        Recorre recursivamente el directorio o fichero "root" e indexa su contenido.
        Los argumentos adicionales "**args" son necesarios para las funcionalidades ampliadas.
        """
        self.positional = args['positional']
        self.semantic = args['semantic']

        file_or_dir = Path(root)
        if file_or_dir.is_file():
            self.index_file(root)
        elif file_or_dir.is_dir():
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR: {root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        # Una vez procesados todos los ficheros, se construye el KDTree (si se ha activado la búsqueda semántica)
        if self.semantic is True:
            # Carga el modelo si no se ha cargado
            self.load_semantic_model()
            if self.chunks:
                self.create_kdtree()
                # Guardamos create_kdtree()
                self.save_info('mi_indice_semantic.pkl')
            else:
                print("Warning: No hay chunks disponibles para construir el KDTree.", file=sys.stderr)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
         
         
            
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
        article.pop('sections')
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article



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
            
            # Si se activa el índice semántico generamos los chunks y embeddings
            if self.semantic:
                self.update_chunks(article['all'], artid)
                        
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
        
        
        
    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()



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

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def solve_query(self, query: str, prev: Dict = {}):
        """
        Procesa una consulta booleana o semántica y devuelve los artículos que la satisfacen.

        Args:
            query (str): Consulta a procesar.
            prev (Dict): Parámetro reservado para futuras ampliaciones (no se usa actualmente).

        Returns:
            Tuple[List[int], List]: Lista de identificadores de artículos que cumplen la consulta y una lista vacía (para posibles errores).
        """

        # Normaliza la consulta a minúsculas
        query = query.lower()
        
        # Si está activada la búsqueda semántica, delega la consulta al motor semántico
        if self.semantic:
            semantic_results = self.solve_semantic_query(query)
            # Si además está activado el reranking semántico, reordena los resultados
            '''
            if self.semantic_ranking:
                semantic_results = self.semantic_reranking(query, semantic_results)
            return semantic_results, []
            '''
            
        # Si la consulta está vacía, no hay resultados que devolver
        if not query:
            return [], []

        terms = []  # Lista de términos parseados de la consulta (pueden ser palabras, frases o NOT)
        i = 0
        length = len(query)
        
        # --- Fase 1: Parseo de la consulta ---
        while i < length:
            # Ignora espacios en blanco
            while i < length and query[i] == ' ':
                i += 1
            if i >= length:
                break

            # Detecta el operador NOT
            if query[i:i+3].upper() == 'NOT':
                terms.append(('NOT', None))
                i += 3
                continue

            # Detecta frases entre comillas
            if query[i] == '"':
                j = i + 1
                while j < length and query[j] != '"':
                    j += 1
                phrase = query[i+1:j]
                terms.append(('PHRASE', self.tokenize(phrase)))  # Tokeniza la frase completa
                i = j + 1
            else:
                # Detecta términos individuales (palabras)
                j = i
                while j < length and query[j] not in (' ', '"'):
                    j += 1
                term = query[i:j]
                if term.upper() == 'NOT':
                    terms.append(('NOT', None))
                else:
                    # Tokeniza el término individual y lo añade
                    tokens = self.tokenize(term)
                    if tokens:
                        terms.append(('TERM', tokens[0]))
                i = j

        result = None  # Conjunto de artículos que cumplen la consulta

        # --- Fase 2: Evaluación de la consulta ---
        i = 0
        while i < len(terms):
            kind, value = terms[i]

            if kind == 'NOT':
                # El operador NOT debe ir seguido de un término o frase
                if i+1 >= len(terms):
                    raise ValueError("NOT sin término a negar")
                
                next_kind, next_value = terms[i+1]

                # Obtiene los artículos que contienen el término/frase a negar
                if next_kind == 'PHRASE':
                    posting = self.get_positionals(next_value)
                elif next_kind == 'TERM':
                    posting = self.get_posting(next_value)
                else:
                    raise ValueError("Elemento inválido después de NOT")

                # El universo es el conjunto de artículos actuales o todos si es el primer término
                universe = set(result) if result is not None else set(self.articles.keys())
                negated = sorted(universe - set(posting))  # Elimina los artículos que contienen el término/frase

                # Actualiza el resultado acumulado
                result = negated

                i += 2  # Salta el NOT y el término/frase siguiente
            else:
                # Obtiene los artículos que contienen el término o la frase
                if kind == 'PHRASE':
                    posting = self.get_positionals(value)
                elif kind == 'TERM':
                    posting = self.get_posting(value)
                else:
                    posting = []

                # Si es el primer término, inicializa el resultado; si no, hace intersección (AND)
                if result is None:
                    result = posting
                else:
                    result = self.and_posting(result, posting)

                i += 1

        # Si está activado el reranking semántico, reordena los resultados obtenidos por consulta booleana
        if self.semantic_ranking and result:
            result = self.semantic_reranking(query, result)

        # Devuelve la lista de artículos que cumplen la consulta y una lista vacía (para posibles errores)
        return result if result is not None else [], []



    def get_posting(self, term: str):
        # Comprueba si el término está en el índice invertido
        if term in self.index:
            # Devuelve una lista de identificadores de documento donde aparece el término.
            # self.index[term] es una lista de tuplas (docid, posiciones)
            return [docid for docid, _ in self.index[term]]
        
        # Si el término no está en el índice, se devuelve una lista vacía
        return []



    def get_positionals(self, terms: list):
        # Si no se recibe ninguna lista de términos, no hay nada que buscar
        if not terms:
            return []
        
        # Si algún término no está en el índice, no puede haber coincidencias de frase
        for term in terms:
            if term not in self.index:
                return []

        # Inicializar un diccionario con los documentos y posiciones del primer término
        # artid_positions[doc_id] = conjunto de posiciones donde aparece el primer término
        artid_positions = {}
        for artid, positions in self.index[terms[0]]:
            artid_positions[artid] = set(positions)

        # Para cada término siguiente, comprobar si aparece justo después del anterior
        for i in range(1, len(terms)):
            term = terms[i]
            current_positions = {}  # Guardará las nuevas posiciones válidas por documento

            for artid, positions in self.index[term]:
                if artid in artid_positions:
                    prev_pos = artid_positions[artid]
                    
                    # Nos quedamos con las posiciones donde el término actual aparece
                    # justo después de una aparición del término anterior (posición - 1)
                    current_pos = {p for p in positions if (p - 1) in prev_pos}

                    # Si hay coincidencias válidas, las guardamos
                    if current_pos:
                        current_positions[artid] = current_pos
            
            # Actualizamos las posiciones válidas para el siguiente término
            artid_positions = current_positions

        # Devolvemos la lista ordenada de documentos donde se ha encontrado la frase completa
        return sorted(artid_positions.keys())



    def reverse_posting(self, p: list):
        # Obtener el conjunto de todos los IDs de artículos disponibles en la colección
        all_artids = set(self.articles.keys())

        # Calcular la diferencia: documentos que no están en la lista p
        # Es decir, aquellos que NO contienen el término consultado
        return sorted(all_artids - set(p))



    def and_posting(self, p1: list, p2: list):
        i, j = 0, 0           # Inicializamos índices para recorrer ambas listas
        result = []           # Lista donde almacenaremos la intersección (documentos comunes)

        # Mientras no lleguemos al final de ninguna lista
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                # Si ambos documentos coinciden, los añadimos al resultado
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                # Si el documento en p1 es menor, avanzamos en p1 para buscar coincidencia
                i += 1
            else:
                # Si el documento en p2 es menor, avanzamos en p2
                j += 1

        return result  # Devolvemos la lista ordenada de documentos comunes



    def minus_posting(self, p1: list, p2: list):
        result = []      # Lista donde guardaremos documentos que están en p1 pero no en p2
        i, j = 0, 0     # Índices para recorrer ambas listas ordenadas

        # Recorremos toda la lista p1
        while i < len(p1):
            # Si ya hemos terminado p2 o el documento actual de p1 es menor que el de p2
            # significa que ese documento de p1 no está en p2, por tanto lo añadimos
            if j >= len(p2) or p1[i] < p2[j]:
                result.append(p1[i])
                i += 1
            # Si los documentos son iguales, los saltamos (no se añaden a resultado)
            elif p1[i] == p2[j]:
                i += 1
                j += 1
            # Si el documento de p2 es menor, avanzamos en p2 para buscar coincidencias
            else:
                j += 1

        return result  # Devolvemos la lista de documentos que están en p1 pero no en p2

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
                # INICIO CAMBIO EN v1.1
                result, _ = self.solve_query(query)
                result = len(result)
                # FIN CAMBIO EN v1.1
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors



    def solve_and_show(self, query: str):
        """
        Ejecuta la consulta y muestra los resultados con el siguiente formato:
        
        ========================================
        # 01 (258) Mundo sublunar:    https://es.wikipedia.org/wiki/Mundo_sublunar
        # 02 (134) Exposición Especializada de Zaragoza (2008):    https://es.wikipedia.org/wiki/Exposici%C3%B3n_Internacional_de_Zaragoza_(2008)
        ...
        ========================================
        Number of results: 17

        Se calcula el "score" como el número de tokens que contiene el campo "all" del artículo.
        """
        # Ejecuta la consulta; se asume que solve_query devuelve una tupla,
        # donde el primer elemento es la lista de artids (los resultados)
        result, _ = self.solve_query(query)
        total_results = len(result)

        # Recorremos cada resultado para obtener la información del artículo
        display_results = []
        for artid in result:
            # Obtiene (docid, línea) del artículo
            docid, line = self.articles[artid]
            article_file = self.docs[docid]
            
            # Abre el fichero y localiza la línea donde se encuentra el artículo
            with open(article_file, encoding="utf-8") as fh:
                for idx, raw in enumerate(fh):
                    if idx == line:
                        article = self.parse_article(raw)
                        break
            
            # Calcula un "score" simple basado en la cantidad de tokens del artículo (campo "all")
            score = len(article.get("all", "").split())
            title = article.get("title", "Título desconocido")
            url = article.get("url", "URL desconocida")
            
            display_results.append((artid, score, title, url))
        
        # Determina el número de resultados a mostrar: todos si show_all es True, o hasta SHOW_MAX
        max_results = total_results if self.show_all else self.SHOW_MAX

        # Encabezado de salida
        print("========================================")
        # Muestra los primeros max_results con un formato numerado
        for idx, (artid, score, title, url) in enumerate(display_results[:max_results], start=1):
            print(f"# {idx:02d} ({score:3d}) {title}:\t{url}")
        print("========================================")
        print(f"Number of results: {total_results}")