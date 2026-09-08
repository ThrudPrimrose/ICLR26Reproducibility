subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(C, name="tsvc_2_s316_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value, intent(in) :: a, result
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), dimension(:), pointer :: aarr => null()
  real(c_double), pointer :: rptr => null()
  real(c_double) :: x, t0, t1
  integer(c_int64_t) :: i, n
  integer :: kt, k

  call C_F_POINTER(a, aarr, [LEN_1D])
  call C_F_POINTER(result, rptr)
  n = LEN_1D
  do k = 1, 9
     kt = 2**(k-1)   ! 1 2 4 8 16 32 64 128 256
     call omp_set_num_threads(kt)
     x = aarr(1)
     t0 = omp_get_wtime()
     !$omp parallel do reduction(min:x)
     do i = 1, n
        if (aarr(i) < x) x = aarr(i)
     end do
     !$omp end parallel do
     t1 = omp_get_wtime()
     write(*,'(A,I0,A,F8.3,A,F7.1,A,F10.5)') 'PROBE scalar threads=', kt, ' ms=', t1-t0, ' GB/s=', (real(n)*8.0d0)/((t1-t0)*1.0d9), ' min=', x
  end do
  flush(6)
  rptr = x
end subroutine tsvc_2_s316_fp64
