// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper


#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

#ifndef NEL
#define NEL 97046
#endif
#ifndef NELR
#define NELR 97280
#endif

#ifdef _OPENARC_
#if NEL == 97046
#pragma openarc #define NEL 97046
#elif NEL == 193474
#pragma openarc #define NEL 193474
#elif NEL == 232536
#pragma openarc #define NEL 232536
#endif

#if NELR == 97280
#pragma openarc #define NELR 97280
#elif NELR == 97152
#pragma openarc #define NELR 97152
#elif NELR == 97088
#pragma openarc #define NELR 97088
#elif NELR == 97056
#pragma openarc #define NELR 97056
#elif NELR == 193536
#pragma openarc #define NELR 193536
#elif NELR == 193504
#pragma openarc #define NELR 193504
#elif NELR == 193474
#pragma openarc #define NELR 193474
#elif NELR == 232536
#pragma openarc #define NELR 232536
#endif

#endif

struct float3S { float x, y, z; };

#ifndef block_length
#ifdef _OPENMP
#error "you need to define block_length"
#else
#define block_length 1
#endif
#endif

/*
 * Options
 *
 */
#define GAMMA 1.4F
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2F
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

#ifdef _OPENARC_
#pragma openarc #define NDIM 3
#pragma openarc #define NNB 4
#pragma openarc #define VAR_DENSITY 0
#pragma openarc #define VAR_MOMENTUM  1
#pragma openarc #define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#pragma openarc #define NVAR (VAR_DENSITY_ENERGY+1)
#endif

#define min(a,b) ((a<b)?a:b)

double gettime() {
		struct timeval t;
		gettimeofday(&t,0);
		return t.tv_sec+t.tv_usec*1e-6;
}


float* alloc_float(int N)
{
		return (float *)malloc(N * sizeof(float));
}

void dump(float* variables, int nel, int nelr)
{


		int i, j;
		FILE *fp;
		if( (fp = fopen("density", "w")) == NULL ) {
				printf("Can not open %s\n", "density");
				exit(1);
		}
		fprintf(fp, "%d %d\n", nel, nelr);
		for(i = 0; i < nel; i++) {
				fprintf(fp, "%f\n", variables[i + VAR_DENSITY*nelr]);
		} 
		fclose(fp);


		if( (fp = fopen("momentum", "w")) == NULL ) {
				printf("Can not open %s\n", "momentum");
				exit(1);
		}
		fprintf(fp, "%d %d\n", nel, nelr);
		for(i = 0; i < nel; i++) {
				for(j = 0; j != NDIM; j++) {
						fprintf(fp, "%g ", variables[i + (VAR_MOMENTUM+j)*nelr]);
				}
				fprintf(fp, "\n");
		} 
		fclose(fp);

		if( (fp = fopen("density_energy", "w")) == NULL ) {
				printf("Can not open %s\n", "density_energy");
				exit(1);
		}
		fprintf(fp, "%d %d\n", nel, nelr);
		for(i = 0; i < nel; i++) {
				fprintf(fp, "%f\n", variables[i + VAR_DENSITY_ENERGY*nelr]);
		} 
		fclose(fp);

}

/*
 * Element-based Cell-centered FVM solver functions
 */
float ff_variable[NVAR];
//struct float3S ff_flux_contribution_momentum_x;
float ff_flux_contribution_momentum_x_x;
float ff_flux_contribution_momentum_x_y;
float ff_flux_contribution_momentum_x_z;
//struct float3S ff_flux_contribution_momentum_y;
float ff_flux_contribution_momentum_y_x;
float ff_flux_contribution_momentum_y_y;
float ff_flux_contribution_momentum_y_z;
//struct float3S ff_flux_contribution_momentum_z;
float ff_flux_contribution_momentum_z_x;
float ff_flux_contribution_momentum_z_y;
float ff_flux_contribution_momentum_z_z;
//struct float3S ff_flux_contribution_density_energy;
float ff_flux_contribution_density_energy_x;
float ff_flux_contribution_density_energy_y;
float ff_flux_contribution_density_energy_z;

void verify(float* variables, int nel, int nelr) 
{

        int i, j;
        FILE *fp;
        double deltaL2Norm = 0;
        double nonAccL2Norm = 0;
        double L2Norm;
        char line [ 128 ];  
        float d;
        float cpuVal;
        FILE *file;
        char *filename = "density97k";
        if(nel != 97046) {
            printf("Not verifying: This verification only works with datasize of 97k. You have specified an incorrect NEL %d\n", nel);
            return;
        }   
    
    
        file = fopen ( filename, "r" );
        if ( file != NULL )
        {   
    
            fgets ( line, sizeof line, file );
            i=0;
            while ( fgets ( line, sizeof line, file ) != NULL ) 
            {   
                //fputs ( line, stdout ); /* write the line */
              cpuVal = atof(line);
                d = cpuVal - variables[i + VAR_DENSITY*nelr];
                deltaL2Norm += d * d;
                nonAccL2Norm += cpuVal* cpuVal;
                i++;
            }   
            fclose ( file );
            L2Norm = sqrt(deltaL2Norm / nonAccL2Norm);
    
            if (L2Norm < 1e-6)
                printf("Verification: Successful\n");
            else
                printf("Verification: Failed\n");   
        }   
        else
        {   
            printf("Error : Could not find the ideal output file 'density97k' (CPU output)");
        }   
    
}



/*
 * Main function
 */
int main(int argc, char* argv[])
{
		double start_time, end_time;
		double start_time1, end_time1;
		double start_time2;
		int i, j, k;
		int nel;
		int nelr;
		float* areas;
		int* elements_surrounding_elements;
		float* normals;
		float* variables;
		float* old_variables;
		float* fluxes;
		float* step_factors;
		const char* data_file_name;
		const float angle_of_attack = ((float)(3.1415926535897931 / 180.0f)) * ((float)deg_angle_of_attack);
		float ff_pressure;
		float ff_speed_of_sound;
		float ff_speed;
		//struct float3S ff_velocity;
		float ff_velocity_x;
		float ff_velocity_y;
		float ff_velocity_z;
		//struct float3S ff_momentum;
		float ff_momentum_x;
		float ff_momentum_y;
		float ff_momentum_z;
		FILE * fp;
		int last;

		/////////////////////
		// Used for copy() //
		/////////////////////
		int i_c;

		////////////////////////////////////
		// Used for compute_step_factor() //
		////////////////////////////////////
		int i_csf;
		float density;
		//struct float3S momentum;
		float momentum_x;
		float momentum_y;
		float momentum_z;
		float density_energy;
		//struct float3S velocity;	   
		float velocity_x;
		float velocity_y;
		float velocity_z;
		float speed_sqd;
		float pressure;
		float speed_of_sound;

		/////////////////////////////
		// Used for compute_flux() //
		/////////////////////////////
		const float smoothing_coefficient = 0.2f;
		int i_cf;
		int j_cf, nb;
		//struct float3S normal; float normal_len;
		float normal_x; 
		float normal_y; 
		float normal_z; 
		float normal_len;
		float factor;
		float density_i;
		//struct float3S momentum_i;
		float momentum_i_x;
		float momentum_i_y;
		float momentum_i_z;
		float density_energy_i;
		//struct float3S velocity_i;             				 
		float velocity_i_x;
		float velocity_i_y;
		float velocity_i_z;
		float speed_sqd_i;
		float speed_i;
		float pressure_i;
		float speed_of_sound_i;
		//struct float3S flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
		float flux_contribution_i_momentum_x_x;
		float flux_contribution_i_momentum_x_y;
		float flux_contribution_i_momentum_x_z;
		float flux_contribution_i_momentum_y_x;
		float flux_contribution_i_momentum_y_y;
		float flux_contribution_i_momentum_y_z;
		float flux_contribution_i_momentum_z_x;
		float flux_contribution_i_momentum_z_y;
		float flux_contribution_i_momentum_z_z;
		//struct float3S flux_contribution_i_density_energy;
		float flux_contribution_i_density_energy_x;
		float flux_contribution_i_density_energy_y;
		float flux_contribution_i_density_energy_z;
		float flux_i_density;
		//struct float3S flux_i_momentum;
		float flux_i_momentum_x;
		float flux_i_momentum_y;
		float flux_i_momentum_z;
		float flux_i_density_energy;
		//struct float3S velocity_nb;
		float velocity_nb_x;
		float velocity_nb_y;
		float velocity_nb_z;
		float density_nb, density_energy_nb;
		//struct float3S momentum_nb;
		float momentum_nb_x;
		float momentum_nb_y;
		float momentum_nb_z;
		//struct float3S flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
		float flux_contribution_nb_momentum_x_x;
		float flux_contribution_nb_momentum_x_y;
		float flux_contribution_nb_momentum_x_z;
		float flux_contribution_nb_momentum_y_x;
		float flux_contribution_nb_momentum_y_y;
		float flux_contribution_nb_momentum_y_z;
		float flux_contribution_nb_momentum_z_x;
		float flux_contribution_nb_momentum_z_y;
		float flux_contribution_nb_momentum_z_z;
		//struct float3S flux_contribution_nb_density_energy;
		float flux_contribution_nb_density_energy_x;
		float flux_contribution_nb_density_energy_y;
		float flux_contribution_nb_density_energy_z;
		float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
		float de_p;

		/////////////////////////////
		// Used for time_step() //
		/////////////////////////////
		int i_ts;
		//float factor;

		if (argc < 2)
		{
				printf("specify data file name\n");
				return 0;
		}
		data_file_name = argv[1];

		start_time = gettime();

		// set far field conditions
		{
				ff_variable[VAR_DENSITY] = 1.4f;

				ff_pressure = 1.0f;
				ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
				ff_speed = ((float)ff_mach)*ff_speed_of_sound;

				ff_velocity_x = ff_speed*((float)(cos((float)angle_of_attack)));
				ff_velocity_y = ff_speed*((float)(sin((float)angle_of_attack)));
				ff_velocity_z = 0.0f;

				ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity_x;
				ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity_y;
				ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity_z;

				ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(0.5f*(ff_speed*ff_speed)) + (ff_pressure / ((float)(GAMMA-1.0f)));

				ff_momentum_x = *(ff_variable+VAR_MOMENTUM+0);
				ff_momentum_y = *(ff_variable+VAR_MOMENTUM+1);
				ff_momentum_z = *(ff_variable+VAR_MOMENTUM+2);
				//compute_flux_contribution(ff_variable[VAR_DENSITY], &ff_momentum, ff_variable[VAR_DENSITY_ENERGY], ff_pressure, &ff_velocity, &ff_flux_contribution_momentum_x, &ff_flux_contribution_momentum_y, &ff_flux_contribution_momentum_z, &ff_flux_contribution_density_energy);
				ff_flux_contribution_momentum_x_x = ff_velocity_x*ff_momentum_x + ff_pressure;
				ff_flux_contribution_momentum_x_y = ff_velocity_x*ff_momentum_y;
				ff_flux_contribution_momentum_x_z = ff_velocity_x*ff_momentum_z;

				ff_flux_contribution_momentum_y_x = ff_flux_contribution_momentum_x_y;
				ff_flux_contribution_momentum_y_y = ff_velocity_y*ff_momentum_y + ff_pressure;
				ff_flux_contribution_momentum_y_z = ff_velocity_y*ff_momentum_z;

				ff_flux_contribution_momentum_z_x = ff_flux_contribution_momentum_x_z;
				ff_flux_contribution_momentum_z_y = ff_flux_contribution_momentum_y_z;
				ff_flux_contribution_momentum_z_z = ff_velocity_z*ff_momentum_z + ff_pressure;

				de_p = ff_variable[VAR_DENSITY_ENERGY]+ff_pressure;
				ff_flux_contribution_density_energy_x = ff_velocity_x*de_p;
				ff_flux_contribution_density_energy_y = ff_velocity_y*de_p;
				ff_flux_contribution_density_energy_z = ff_velocity_z*de_p;
		}


		// read in domain geometry
		{
				if( (fp = fopen(data_file_name, "r")) == NULL ) {
						printf("Can not open %s\n", data_file_name);
						exit(1);
				}

				fscanf(fp, "%d", &nel);
				nelr = block_length*((nel / block_length )+ min(1, nel % block_length));
				if( (nel != NEL) || (nelr != NELR) ) {
						printf("NEL or NELR is undefined or has wrong value!\n");
						printf("#define NEL %d\n", nel);
						printf("#define NELR %d\n", nelr);
						exit(1);
				}

				areas =(float *)malloc(sizeof(float) * nelr);
				elements_surrounding_elements = (int *)malloc(sizeof(int)*nelr*NNB);
				normals = (float *)malloc(sizeof(float)*NDIM*NNB*nelr);

				// read in data
				for(i = 0; i < nel; i++)
				{
						fscanf(fp, "%f", &(areas[i]));
						for(j = 0; j < NNB; j++)
						{
								fscanf(fp, "%d", &(elements_surrounding_elements[i + j*nelr]));
								if(elements_surrounding_elements[i+j*nelr] < 0) elements_surrounding_elements[i+j*nelr] = -1;
								elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering

								for(k = 0; k < NDIM; k++)
								{
										fscanf(fp, "%f", &(normals[i + (j + k*NNB)*nelr]));
										normals[i + (j + k*NNB)*nelr] = -normals[i + (j + k*NNB)*nelr];
								}
						}
				}

				// fill in remaining data
				last = nel-1;
				for(i = nel; i < nelr; i++)
				{
						areas[i] = areas[last];
						for(j = 0; j < NNB; j++)
						{
								// duplicate the last element
								elements_surrounding_elements[i + j*nelr] = elements_surrounding_elements[last + j*nelr];
								for(k = 0; k < NDIM; k++) normals[i + (j + k*NNB)*nelr] = normals[last + (j + k*NNB)*nelr];
						}
				}
		}

		variables = alloc_float(nelr*NVAR);
		old_variables = alloc_float(nelr*NVAR);
		fluxes = alloc_float(nelr*NVAR);
		step_factors = alloc_float(nelr);
		start_time2 = gettime();
#pragma acc data copyout(variables[0:NELR*NVAR]) \
        copyin(ff_variable[0:NVAR], areas[0:NELR]) \
        copyin(elements_surrounding_elements[0:NELR*NNB], normals[0:NDIM*NNB*NELR]) \
        create(fluxes[0:NELR*NVAR]) \
        create(old_variables[0:NELR*NVAR], step_factors[0:NELR])
		{
				// Create arrays and set initial conditions
				//initialize_variables(nelr, variables);
#pragma acc kernels loop gang worker independent
				for(i = 0; i < nelr; i++)
				{
#pragma acc loop seq
						for(j = 0; j < NVAR; j++) variables[i + j*nelr] = ff_variable[j];
				}


				// these need to be computed the first time in order to compute time step
				printf("Starting...\n");
				start_time1 = gettime();
				// Begin iterations
				for(i = 0; i < iterations; i++)
				{
						//copy(old_variables, variables, nelr*NVAR);
#pragma acc kernels loop gang worker independent 
						for(i_c = 0; i_c < nelr*NVAR; i_c++)
						{
								old_variables[i_c] = variables[i_c];
						}

						// for the first iteration we compute the time step
						//compute_step_factor(nelr, variables, areas, step_factors);
#pragma acc kernels loop gang worker independent 
						for(i_csf = 0; i_csf < nelr; i_csf++)
						{
								density = variables[i_csf + VAR_DENSITY*nelr];

								momentum_x = variables[i_csf + (VAR_MOMENTUM+0)*nelr];
								momentum_y = variables[i_csf + (VAR_MOMENTUM+1)*nelr];
								momentum_z = variables[i_csf + (VAR_MOMENTUM+2)*nelr];

								density_energy = variables[i_csf + VAR_DENSITY_ENERGY*nelr];
								//compute_velocity(density, &momentum, &velocity);
								velocity_x = momentum_x / density;
								velocity_y = momentum_y / density;
								velocity_z = momentum_z / density;

								//speed_sqd      = compute_speed_sqd(&velocity);
								speed_sqd = velocity_x*velocity_x + velocity_y*velocity_y + velocity_z*velocity_z;
								//pressure       = compute_pressure(density, density_energy, speed_sqd);
								pressure = (((float)GAMMA)-1.0f)*(density_energy - 0.5f*density*speed_sqd);
								//speed_of_sound = compute_speed_of_sound(density, pressure);
								speed_of_sound = sqrtf(((float)GAMMA)*pressure/density);

								// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
								step_factors[i_csf] = 0.5f / (sqrtf(areas[i_csf]) * (sqrtf(speed_sqd) + speed_of_sound));
						}

						for(j = 0; j < RK; j++)
						{
								//compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);

#pragma acc kernels loop gang worker independent 
								for(i_cf = 0; i_cf < nelr; i_cf++)
								{
										density_i = variables[i_cf + VAR_DENSITY*nelr];
										momentum_i_x = variables[i_cf + (VAR_MOMENTUM+0)*nelr];
										momentum_i_y = variables[i_cf + (VAR_MOMENTUM+1)*nelr];
										momentum_i_z = variables[i_cf + (VAR_MOMENTUM+2)*nelr];

										density_energy_i = variables[i_cf + VAR_DENSITY_ENERGY*nelr];

										//compute_velocity(density_i, &momentum_i, &velocity_i);
										velocity_i_x = momentum_i_x / density_i;
										velocity_i_y = momentum_i_y / density_i;
										velocity_i_z = momentum_i_z / density_i;
										//speed_sqd_i = compute_speed_sqd(&velocity_i);
										speed_sqd_i = velocity_i_x*velocity_i_x + velocity_i_y*velocity_i_y + velocity_i_z*velocity_i_z;
										speed_i = sqrtf(speed_sqd_i);
										//pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
										pressure_i = (((float)GAMMA)-1.0f)*(density_energy_i - 0.5f*density_i*speed_sqd_i);
										//speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
										speed_of_sound_i = sqrtf(((float)GAMMA)*pressure_i/density_i);
										//compute_flux_contribution(density_i, &momentum_i, density_energy_i, pressure_i, &velocity_i, &flux_contribution_i_momentum_x, &flux_contribution_i_momentum_y, &flux_contribution_i_momentum_z, &flux_contribution_i_density_energy);
										flux_contribution_i_momentum_x_x = velocity_i_x*momentum_i_x + pressure_i;
										flux_contribution_i_momentum_x_y = velocity_i_x*momentum_i_y;
										flux_contribution_i_momentum_x_z = velocity_i_x*momentum_i_z;

										flux_contribution_i_momentum_y_x = flux_contribution_i_momentum_x_y;
										flux_contribution_i_momentum_y_y = velocity_i_y*momentum_i_y + pressure_i;
										flux_contribution_i_momentum_y_z = velocity_i_y*momentum_i_z;

										flux_contribution_i_momentum_z_x = flux_contribution_i_momentum_x_z;
										flux_contribution_i_momentum_z_y = flux_contribution_i_momentum_y_z;
										flux_contribution_i_momentum_z_z = velocity_i_z*momentum_i_z + pressure_i;

										de_p = density_energy_i+pressure_i;
										flux_contribution_i_density_energy_x = velocity_i_x*de_p;
										flux_contribution_i_density_energy_y = velocity_i_y*de_p;
										flux_contribution_i_density_energy_z = velocity_i_z*de_p;


										flux_i_density = 0.0f;
										flux_i_momentum_x = 0.0f;
										flux_i_momentum_y = 0.0f;
										flux_i_momentum_z = 0.0f;
										flux_i_density_energy = 0.0f;

										for(j_cf = 0; j_cf < NNB; j_cf++)
										{
												nb = elements_surrounding_elements[i_cf + j_cf*nelr];
												normal_x = normals[i_cf + (j_cf + 0*NNB)*nelr];
												normal_y = normals[i_cf + (j_cf + 1*NNB)*nelr];
												normal_z = normals[i_cf + (j_cf + 2*NNB)*nelr];
												normal_len = sqrtf(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z);

												if(nb >= 0) 	// a legitimate neighbor
												{
														density_nb =        variables[nb + VAR_DENSITY*nelr];
														momentum_nb_x =     variables[nb + (VAR_MOMENTUM+0)*nelr];
														momentum_nb_y =     variables[nb + (VAR_MOMENTUM+1)*nelr];
														momentum_nb_z =     variables[nb + (VAR_MOMENTUM+2)*nelr];
														density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
														//compute_velocity(density_nb, &momentum_nb, &velocity_nb);
														velocity_nb_x = momentum_nb_x / density_nb;
														velocity_nb_y = momentum_nb_y / density_nb;
														velocity_nb_z = momentum_nb_z / density_nb;
														//speed_sqd_nb                      = compute_speed_sqd(&velocity_nb);
														speed_sqd_nb = velocity_nb_x*velocity_nb_x + velocity_nb_y*velocity_nb_y + velocity_nb_z*velocity_nb_z;
														//pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
														pressure_nb = (((float)GAMMA)-1.0f)*(density_energy_nb - 0.5f*density_nb*speed_sqd_nb);
														//speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
														speed_of_sound_nb = sqrtf(((float)GAMMA)*pressure_nb/density_nb);
														//compute_flux_contribution(density_nb, &momentum_nb, density_energy_nb, pressure_nb, &velocity_nb, &flux_contribution_nb_momentum_x, &flux_contribution_nb_momentum_y, &flux_contribution_nb_momentum_z, &flux_contribution_nb_density_energy);
														flux_contribution_nb_momentum_x_x = velocity_nb_x*momentum_nb_x + pressure_nb;
														flux_contribution_nb_momentum_x_y = velocity_nb_x*momentum_nb_y;
														flux_contribution_nb_momentum_x_z = velocity_nb_x*momentum_nb_z;

														flux_contribution_nb_momentum_y_x = flux_contribution_nb_momentum_x_y;
														flux_contribution_nb_momentum_y_y = velocity_nb_y*momentum_nb_y + pressure_nb;
														flux_contribution_nb_momentum_y_z = velocity_nb_y*momentum_nb_z;

														flux_contribution_nb_momentum_z_x = flux_contribution_nb_momentum_x_z;
														flux_contribution_nb_momentum_z_y = flux_contribution_nb_momentum_y_z;
														flux_contribution_nb_momentum_z_z = velocity_nb_z*momentum_nb_z + pressure_nb;

														de_p = density_energy_nb+pressure_nb;
														flux_contribution_nb_density_energy_x = velocity_nb_x*de_p;
														flux_contribution_nb_density_energy_y = velocity_nb_y*de_p;
														flux_contribution_nb_density_energy_z = velocity_nb_z*de_p;

														// artificial viscosity
														factor = -normal_len*smoothing_coefficient*0.5f*(speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
														flux_i_density += factor*(density_i-density_nb);
														flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
														flux_i_momentum_x += factor*(momentum_i_x-momentum_nb_x);
														flux_i_momentum_y += factor*(momentum_i_y-momentum_nb_y);
														flux_i_momentum_z += factor*(momentum_i_z-momentum_nb_z);

														// accumulate cell-centered fluxes
														factor = 0.5f*normal_x;
														flux_i_density += factor*(momentum_nb_x+momentum_i_x);
														flux_i_density_energy += factor*(flux_contribution_nb_density_energy_x+flux_contribution_i_density_energy_x);
														flux_i_momentum_x += factor*(flux_contribution_nb_momentum_x_x+flux_contribution_i_momentum_x_x);
														flux_i_momentum_y += factor*(flux_contribution_nb_momentum_y_x+flux_contribution_i_momentum_y_x);
														flux_i_momentum_z += factor*(flux_contribution_nb_momentum_z_x+flux_contribution_i_momentum_z_x);

														factor = 0.5f*normal_y;
														flux_i_density += factor*(momentum_nb_y+momentum_i_y);
														flux_i_density_energy += factor*(flux_contribution_nb_density_energy_y+flux_contribution_i_density_energy_y);
														flux_i_momentum_x += factor*(flux_contribution_nb_momentum_x_y+flux_contribution_i_momentum_x_y);
														flux_i_momentum_y += factor*(flux_contribution_nb_momentum_y_y+flux_contribution_i_momentum_y_y);
														flux_i_momentum_z += factor*(flux_contribution_nb_momentum_z_y+flux_contribution_i_momentum_z_y);

														factor = 0.5f*normal_z;
														flux_i_density += factor*(momentum_nb_z+momentum_i_z);
														flux_i_density_energy += factor*(flux_contribution_nb_density_energy_z+flux_contribution_i_density_energy_z);
														flux_i_momentum_x += factor*(flux_contribution_nb_momentum_x_z+flux_contribution_i_momentum_x_z);
														flux_i_momentum_y += factor*(flux_contribution_nb_momentum_y_z+flux_contribution_i_momentum_y_z);
														flux_i_momentum_z += factor*(flux_contribution_nb_momentum_z_z+flux_contribution_i_momentum_z_z);
												}
												else if(nb == -1)	// a wing boundary
												{
														flux_i_momentum_x += normal_x*pressure_i;
														flux_i_momentum_y += normal_y*pressure_i;
														flux_i_momentum_z += normal_z*pressure_i;
												}
												else if(nb == -2) // a far field boundary
												{
														factor = 0.5f*normal_x;
														flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i_x);
														flux_i_density_energy += factor*(ff_flux_contribution_density_energy_x+flux_contribution_i_density_energy_x);
														flux_i_momentum_x += factor*(ff_flux_contribution_momentum_x_x + flux_contribution_i_momentum_x_x);
														flux_i_momentum_y += factor*(ff_flux_contribution_momentum_y_x + flux_contribution_i_momentum_y_x);
														flux_i_momentum_z += factor*(ff_flux_contribution_momentum_z_x + flux_contribution_i_momentum_z_x);

														factor = 0.5f*normal_y;
														flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i_y);
														flux_i_density_energy += factor*(ff_flux_contribution_density_energy_y+flux_contribution_i_density_energy_y);
														flux_i_momentum_x += factor*(ff_flux_contribution_momentum_x_y + flux_contribution_i_momentum_x_y);
														flux_i_momentum_y += factor*(ff_flux_contribution_momentum_y_y + flux_contribution_i_momentum_y_y);
														flux_i_momentum_z += factor*(ff_flux_contribution_momentum_z_y + flux_contribution_i_momentum_z_y);

														factor = 0.5f*normal_z;
														flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i_z);
														flux_i_density_energy += factor*(ff_flux_contribution_density_energy_z+flux_contribution_i_density_energy_z);
														flux_i_momentum_x += factor*(ff_flux_contribution_momentum_x_z + flux_contribution_i_momentum_x_z);
														flux_i_momentum_y += factor*(ff_flux_contribution_momentum_y_z + flux_contribution_i_momentum_y_z);
														flux_i_momentum_z += factor*(ff_flux_contribution_momentum_z_z + flux_contribution_i_momentum_z_z);

												}
										}

										fluxes[i_cf + VAR_DENSITY*nelr] = flux_i_density;
										fluxes[i_cf + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum_x;
										fluxes[i_cf + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum_y;
										fluxes[i_cf + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum_z;
										fluxes[i_cf + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
								}
								//time_step(j, nelr, old_variables, variables, step_factors, fluxes);
#pragma acc kernels loop gang worker independent 
								for(i_ts = 0; i_ts < nelr; i_ts++)
								{
										factor = step_factors[i_ts]/((float)(RK+1-j));

										variables[i_ts + VAR_DENSITY*nelr] = old_variables[i_ts + VAR_DENSITY*nelr] + factor*fluxes[i_ts + VAR_DENSITY*nelr];
										variables[i_ts + VAR_DENSITY_ENERGY*nelr] = old_variables[i_ts + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i_ts + VAR_DENSITY_ENERGY*nelr];
										variables[i_ts + (VAR_MOMENTUM+0)*nelr] = old_variables[i_ts + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i_ts + (VAR_MOMENTUM+0)*nelr];
										variables[i_ts + (VAR_MOMENTUM+1)*nelr] = old_variables[i_ts + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i_ts + (VAR_MOMENTUM+1)*nelr];
										variables[i_ts + (VAR_MOMENTUM+2)*nelr] = old_variables[i_ts + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i_ts + (VAR_MOMENTUM+2)*nelr];
								}

						}
				}
		}

		end_time1 = gettime();
		printf("Accelerator Elapsed Time = %lf sec.\n", end_time1 - start_time1);
        printf("Main Comp. TIme (with data copies) = %lf sec.\n", end_time1 - start_time2);
        if(VERIFICATION) {
            verify(variables, nel, nelr);
        }


		printf("Saving solution...\n");
		dump(variables, nel, nelr);
		printf("Saved solution...\n");


		printf("Cleaning up...\n");
		free((void *)areas);
		free((void *)elements_surrounding_elements);
		free((void *)normals);

		free((void *)variables);
		free((void *)old_variables);
		free((void *)fluxes);
		free((void *)step_factors);

		printf("Done...\n");

		end_time = gettime();
		printf("Total Execution Time = %lf sec.\n", end_time - start_time);

		return 0;
}
