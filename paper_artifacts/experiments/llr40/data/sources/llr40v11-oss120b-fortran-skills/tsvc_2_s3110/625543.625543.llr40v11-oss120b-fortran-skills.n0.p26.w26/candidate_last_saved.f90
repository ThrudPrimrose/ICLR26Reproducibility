subroutine tsvc_2_s3110_fp64(a, b, LEN_2D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(in) :: a(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: b(*)

  integer(c_int64_t) :: i, j
  real(c_double) :: v
  real(c_double) :: global_max, local_max
  integer(c_int64_t) :: global_max_i, global_max_j
  integer(c_int64_t) :: local_i, local_j
  integer(c_int64_t) :: i_idx, local_idx

  ! Initialize global maximum with first element (C[0,0])
  global_max = a(1,1)
  global_max_i = 0_c_int64_t
  global_max_j = 0_c_int64_t

!$omp parallel private(i,j,v,local_max,local_i,local_j,i_idx,local_idx) shared(global_max,global_max_i,global_max_j)
  local_max = -huge(0.0_c_double)
  local_i = -1_c_int64_t
  local_j = -1_c_int64_t

  !$omp do schedule(static)
  do i = 1, LEN_2D
    do j = 1, LEN_2D
      v = a(j,i)   ! map Fortran (j,i) to C (i-1,j-1)
      if (v > local_max) then
        local_max = v
        local_i = i - 1_c_int64_t
        local_j = j - 1_c_int64_t
      else if (v == local_max) then
        i_idx = (i - 1_c_int64_t) * LEN_2D + (j - 1_c_int64_t)
        local_idx = local_i * LEN_2D + local_j
        if (i_idx < local_idx) then
          local_i = i - 1_c_int64_t
          local_j = j - 1_c_int64_t
        end if
      end if
    end do
  end do
  !$omp end do

  !$omp critical
    if (local_max > global_max) then
      global_max = local_max
      global_max_i = local_i
      global_max_j = local_j
    else if (local_max == global_max) then
      if (local_i * LEN_2D + local_j < global_max_i * LEN_2D + global_max_j) then
        global_max_i = local_i
        global_max_j = local_j
      end if
    end if
  !$omp end critical
!$omp end parallel

  b(1) = global_max + real(global_max_i, c_double) + real(global_max_j, c_double)

end subroutine tsvc_2_s3110_fp64
