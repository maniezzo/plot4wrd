using System;
using System.IO;

namespace patchcsv
{
   class Patcher
   {
      public void patch()
      {  string line;
         char[] separators = new char[]{','};
         string[] elem;
         int cont,nattr;

         StreamReader fin = new StreamReader("../20210121_102003_fusion_GC_1.csv");
         StreamWriter fout= new StreamWriter("../fusion_GC_1.csv");
         line = fin.ReadLine();
         fout.WriteLine(line);
         elem = line.Split(separators, StringSplitOptions.RemoveEmptyEntries);
         nattr= elem.Length;
         cont = 1;
         while (fin.Peek() >= 0)
         {
            line = fin.ReadLine();
            elem = line.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            Console.WriteLine(cont+") "+elem.Length+" "+line);
            for(int i=0;i<elem.Length;i++)
            {  fout.Write(elem[i]+",");
               if(i==elem.Length-13)
               {
                  for(int j=0;j<nattr-elem.Length;j++)
                     fout.Write(",");
               }
            }
            fout.WriteLine();
            //fout.WriteLine(line);
            cont++;
         }
         fin.Close();
         fout.Close();

         fin = new StreamReader("../20210121_102501_fusion_EG_1.csv");
         fout= new StreamWriter("../fusion_EG_1.csv");
         line = fin.ReadLine();
         fout.WriteLine(line);
         elem = line.Split(separators, StringSplitOptions.RemoveEmptyEntries);
         nattr= elem.Length;
         cont = 1;
         while (fin.Peek() >= 0)
         {
            line = fin.ReadLine();
            elem = line.Split(separators, StringSplitOptions.RemoveEmptyEntries);
            Console.WriteLine(cont+") "+elem.Length+" "+line);
            for(int i=0;i<elem.Length;i++)
            {  fout.Write(elem[i]+",");
               if(i==elem.Length-13)
               {
                  for(int j=0;j<nattr-elem.Length;j++)
                     fout.Write(",");
               }
            }
            fout.WriteLine();
            //fout.WriteLine(line);
            cont++;
         }
         fin.Close();
         fout.Close();
      }
   }    
   class Program
   {
        static void Main(string[] args)
        {
            Patcher P = new Patcher();
            P.patch();
            Console.WriteLine("Finito");
        }
   }
}
