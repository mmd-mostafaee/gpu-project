function lagout = lagrange(pixelx, pixely, accuracy, matrix)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
%Hossein Ghasemi Ramshe  9433590
%Ali Taheri   9433597

    x = pixelx;
    y = pixely;
    table = [];
    table=int64(table);
    i = 0;
    index = 0;
    x=int64(x);
    while (i <  (accuracy / 2)) && (x>1)
        x = x - 1;
        if rem(x, 4) == 1
            table = [table [x; matrix(y,x)]];
            index = index + 1;
            i = i + 1;
        else
            continue;
        end
        if (i <  (accuracy / 2)) && (x>1)
            ff = '1 ok';
        end
        if i <  (accuracy / 2)
            ff = '2 ok';
        end
        if (x>1)
            ff = '3 ok';
        end
    end
    
    x = pixelx;
    x=int64(x);
    while i <  accuracy && x<1024
        x = x + 1;
        if rem(x, 4) == 1
            table = [table [x; matrix(y,x)]];
            index = index + 1;
            i = i + 1;
        else
            continue
        end
    end
    table = int64(table);
    result=0;
    inp=int64(pixelx);
    for t = 1 : index
        sorat=1;
        makhraj=1;
        for i=1 :index
            if i~= t
                sorat=sorat * ( inp - table(1, i));
            end
        end
        for j=1 : index
            if j~= t
               makhraj=makhraj*(table(1,t)-table(1,j)); 
            end
        end
        
        result=result+double(table(2,t)*(double(sorat)/double(makhraj)));
    end
    lagout=result;
end

