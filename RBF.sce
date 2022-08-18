clc
clear all



// lendo a base de dados
base=csvRead('dermatology.data')



// retirando linhas com dados faltantes
base(isnan(sum(base,2)),:) = []



// ordenando as linhas da base (ordem crescente, de cima para baixo) pelo tipo da doença (coluna 35)
[dummy,idx]=gsort(base(:,35),"g","i") 
base=base(idx,:)



// vec[i] deve indicar a quantidade de elementos i na coluna do tipo da doença (coluna 35)
// há ao todo 6 classes
vec = []
for i=1:6
    [nb, loc] = members([i], base(:,35), "last")
    vec = [vec nb]
    
end



// obtenção da matriz de rótulos já no formato desejado (em cada coluna, o tipo da doença é indicado pelo valor 1, e os outros pelo valor 0)
D = zeros(6,358)     


for i=1:6
    for k=1:vec(i)
     D(i, k+sum(vec(1:i-1))) = 1
    end
end



// dados de entrada no formato desejado
E = base(:,1:34)'    



// normalização por z-score dos dados de entrada
for i=1:34
    E(i,:) = (E(i,:)-mean(E(i, :)))/stdev(E(i,:)) 
end



// aproximadamente metade das amostras de cada classe será usada para treinamento e o resto para teste
// vec = [111 60 71 48 48 20]
// dessa forma, treino: 56 (1), 30(2), 36(3), 24 (4), 24 (5) e 10 (6) -> totalizando 180 para treino
// teste: 55 (1), 30(2), 35(3), 24 (4), 24 (5) e 10 (6) -> totalizando 178 para teste
// os intervalos dos índices estão mostrados a seguir:

ind_treino = [1:56, 112:141, 172:207, 243:266, 291:314, 339:348]
ind_teste = [57:111, 142:171, 208:242, 267:290, 315: 338, 349:358]



// divisão dos dados de entrada entre treino e teste
E_treino = E(:, ind_treino)
E_teste = E(:, ind_teste)



// divisão dos rótulos entre treino e teste
D_treino = D(:,ind_treino)
D_teste = D(:,ind_teste)



[p_treino N_treino] = size(E_treino)



q = 40 // quantidade de neurônios ocultos



T = rand(p_treino,q,'normal')  // vetores centroide dos q neurônios ocultos 



// obtenção de Z_treino
Z_treino = zeros(q,N_treino)


for i=1:N_treino
    for j=1:q
        v = norm(E_treino(:,i) - T(:,j))
        Z_treino(j,i) =exp(-v^2/1000)       // sigma= sqrt(500)
    end
end


Z_treino = [ones(1,N_treino);Z_treino]



// matriz dos pesos dos neurônios de saída
W = D_treino*Z_treino'*(Z_treino*Z_treino')^(-1) 



[p_teste N_teste] = size(E_teste)



// obtenção de Z_teste
Z_teste = zeros(q,N_teste) 


for i=1:N_teste
vetor_teste = E_teste(:,i)
for j=1:q
    v = norm(vetor_teste-T(:,j))
    Z_teste(j,i) = exp(-v^2/1000)          // sigma= sqrt(500)
end

end



Z_teste = [ones(1,N_teste) ; Z_teste]


// Previsão para os dados de teste
Previsao = W*Z_teste



// contador para avaliar a porcentagem de acertos
count = 0
for i=1:N_teste
    
    [a b] = max(D_teste(:,i))
    [c d] = max(Previsao(:,i))
    if b==d
       count = count+1 
    end
    
end
 
 
 
// qtd de acertos dividida pela quantidade de dados de teste (178) e, em seguida, multiplicada por 100
// indica a porcentagem de acertos
disp(100*count/178)











