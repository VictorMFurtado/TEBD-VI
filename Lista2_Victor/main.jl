#=
Este código é referente ao exercício 1 da lista 2 da disciplina Tópicos Especiais em Banco de Dados VI.

Implementar o k-NN
=#

const k = 5

# Mudando a pasta atual do Julia para a pasta deste arquivo
cd(dirname(Base.source_path()))

# Lendo as notas do arquivo
avaliacoes = readdlm("ml-100k/u.data")

# Recupera o maior ID deum usuário
maxIdUsuario = convert(Int32,maximum(avaliacoes[:,1]))
maxIdItem = convert(Int32,maximum(avaliacoes[:,2]))

# Gera a matriz UxI
matriz = zeros(maxIdUsuario,maxIdItem)

for avaliacao = 1:size(avaliacoes,1)
  matriz[avaliacoes[avaliacao,1],avaliacoes[avaliacao,2]] = avaliacoes[avaliacao,3]
end

# Chama o método knn passando alguma função de similaridade como parâmetro
previsao = knn(matriz[1,:],matriz,k,cosseno)
