import faiss
import numpy as np
import pickle
import linecache
import json

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

def get_record_by_line(jsonl_path: str, line_number: int) -> dict:
    """
    使用linecache快速读取指定行号的记录
    """
    line = linecache.getline(jsonl_path, line_number)
    if line.strip():
        return json.loads(line.strip())
    return None


class FaissIndexBuilder:
    def __init__(self, vector_path):
        """
        初始化 FaissIndexBuilder
        Args:
            vector_path (str): .npy 文件路径，存储向量
            use_gpu (bool): 是否使用 GPU 加速
        """
        self.vector_path = vector_path
        self.index = None

    def load_vectors(self):
        """
        加载 .npy 文件中的向量
        """
        # print(f"Loading vectors from {self.vector_path}")
        vectors = np.load(self.vector_path)
        assert len(vectors.shape) == 2, "Vectors should be a 2D numpy array"
        return vectors.astype(np.float32)

    def build_index(self, vectors):
        """
        构建 FAISS 索引
        Args:
            vectors (numpy.ndarray): 2D 向量数组
        """
        dim = vectors.shape[1]
        print(f"Building FAISS index with vectors of dimension {dim}")
        
        # 使用内积作为相似度度量
        index = faiss.IndexFlatIP(dim)
        # index = faiss.IndexFlatL2(dim)

        faiss.normalize_L2(vectors)  # 归一化向量
        index.add(vectors)  # 添加向量到索引

        print(f"FAISS index built with {index.ntotal} entries.")
        self.index = index

    def save_index(self, index_path):
        """
        保存 FAISS 索引到磁盘
        Args:
            index_path (str): 索引保存的路径
        """
        print(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        print("Index saved.")

    def load_index(self, index_path):
        """
        加载 FAISS 索引
        Args:
            index_path (str): 索引文件的路径
        """
        print(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)

        self.index = index
        print("Index loaded.")

    def search(self, query_vector, top_k=10):
        """
        在 FAISS 索引中搜索最近邻
        Args:
            query_vector (numpy.ndarray): 查询向量
            top_k (int): 返回前 k 个最近邻
        Returns:
            D: 距离
            I: 最近邻的索引
        """
        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)  # 归一化查询向量
        D, I = self.index.search(query_vector, top_k)  # 在索引中搜索
        return D, I


if __name__ == "__main__":
    # dataset_embedding_npy为每个dataset item对应embedding，dataset_img_path_pkl为对应每个向量的图片路径
    dataset_embedding_npy = "embedding_train.npy"  # 向量文件路径
    dataset_img_path_json = "image_ids_train.json"
    
    # query_embedding_npy为每个query对应embedding，query_img_path_pkl为对应每个向量的图片路径
    query_embedding_npy = "embeddings_query.npy"
    query_img_path_pkl = "image_ids_query.pkl"

    index_save_path = "index_train.faiss"  # 保存索引的路径
    
    # 获取dataset与query每一条embedding对应的img_path
    # dataset_img_paths = load_pickle_file(dataset_img_path_pkl)
    with open(dataset_img_path_json, "rb") as f:
        dataset_img_paths = json.load(f)

    query_img_paths = load_pickle_file(query_img_path_pkl)

    dataset_json = "database_oven_entity.jsonl"
    query_json = "query_oven_entity.jsonl"

    builder = FaissIndexBuilder(dataset_embedding_npy)

    # 加载向量并构建索引
    dataset_embedding = builder.load_vectors()
    builder.build_index(dataset_embedding)

    # 保存索引
    builder.save_index(index_save_path)

    # 进行查询
    query_embedding = np.load(query_embedding_npy).astype(np.float32)
    # query_embedding = query_embedding[:1903]

    outputs = []

    D, I = builder.search(query_embedding, top_k=20)
    for index in range(len(I)):
        # 对于每个查询，因为query是按顺序一一对应的，所以可以按行读取
        query_item = get_record_by_line(query_json, index+1)
        query_entity_id = query_item['entity_id']
        # query_img_path = query_item['query_image']
        # check
        query_img_path = query_img_paths[index]


        result_entity_ids = []
        result_img_path = []
        distances = []

        for k in range(len(I[index])):
            line_number = I[index][k] + 1
            result_item = get_record_by_line(dataset_json, line_number)

            result_entity_id = result_item['entity_id']
            # top_k_path = result_item['image']
            top_k_path = dataset_img_paths[I[index][k]]
            distance = D[index][k]


            result_img_path.append(top_k_path)
            result_entity_ids.append(result_entity_id)
            distances.append(distance)
        
        # Check Condition 1: Top-2 result_entity_ids are different from query_entity_id
        top_2_diff = (result_entity_ids[0] != query_entity_id) and (result_entity_ids[1] != query_entity_id)

        # Check Condition 2: More than 50% of top-20 result_entity_ids are different from query_entity_id
        total_results = len(result_entity_ids)
        num_diff = sum(1 for eid in result_entity_ids[:total_results] if eid != query_entity_id)
        condition2 = num_diff > (0.5 * total_results)
        if top_2_diff and condition2:
            # Prepare output in JSON format
            output = {}
            output['query'] = {'entity_id': query_entity_id, 'img_path': query_img_path}

            for k in range(total_results):
                output[f'top_{k + 1}'] = {
                    'similarity': float(distances[k]),  # Assuming higher similarity for lower distance
                    'entity_id': result_entity_ids[k],
                    'img_path': result_img_path[k]
                }
            
            # Append the output to the list
            outputs.append(output)
            print(len(outputs))

    # Save all outputs to a JSON file
    with open('query_results_full.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


    # 加载保存的索引
    # builder.load_index(index_save_path)
